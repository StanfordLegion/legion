-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- runs-with:
-- [
--   ["pennant.tests/sedovsmall/sedovsmall.pnt",
--    "-npieces", "1", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-fflow-spmd", "1", "-fvectorize-unsafe", "1", "-ftrace", "0"],
--   ["pennant.tests/sedov/sedov.pnt",
--    "-npieces", "3", "-ll:cpu", "3", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10",
--    "-fflow-spmd", "1", "-fvectorize-unsafe", "1", "-ftrace", "0"],
--   ["pennant.tests/leblanc/leblanc.pnt",
--    "-npieces", "2", "-ll:cpu", "2", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-fflow-spmd", "1", "-fvectorize-unsafe", "1"],
--   ["pennant.tests/sedov/sedov.pnt",
--    "-npieces", "3", "-ll:cpu", "3", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10",
--    "-fflow-spmd", "1", "-fvectorize-unsafe", "1", "-dm:memoize"]
-- ]

-- Inspired by https://github.com/losalamos/PENNANT

import "regent"

require("pennant_common")

sqrt = regentlib.sqrt(double)
fabs = regentlib.fabs(double)

__demand(__inline)
task vec2_add(a : vec2, b : vec2) : vec2
  a.x += b.x
  a.y += b.y
  return a
end

__demand(__inline)
task vec2_sub(a : vec2, b : vec2) : vec2
  a.x -= b.x
  a.y -= b.y
  return a
end

__demand(__inline)
task vec2_neg(a : vec2) : vec2
  a.x = -a.x
  a.y = -a.y
  return a
end

__demand(__inline)
task vec2_mul_lhs(a : double, b : vec2) : vec2
  b.x *= a
  b.y *= a
  return b
end

__demand(__inline)
task vec2_mul_rhs(a : vec2, b : double) : vec2
  a.x *= b
  a.y *= b
  return a
end

__demand(__inline)
task vec2_dot(a : vec2, b : vec2) : double
  return a.x*b.x + a.y*b.y
end

__demand(__inline)
task vec2_cross(a : vec2, b : vec2) : double
  return a.x*b.y - a.y*b.x
end

__demand(__inline)
task vec2_length(a : vec2) : double
  return sqrt(a.x * a.x + a.y * a.y)
end

local c = regentlib.c

-- #####################################
-- ## Initialization
-- #################

-- Hack: This exists to make the compiler recompute the bitmasks for
-- each pointer. This needs to happen here (rather than at
-- initialization time) because we subverted the type system in the
-- construction of the mesh pieces.
task init_pointers(rz : region(zone), rpp : region(point), rpg : region(point),
                   rs : region(side(rz, rpp, rpg, rs)),
                   rs_spans : region(span))
where
  reads writes(rs.{mapsp1, mapsp2}),

  reads(rs_spans)
do
  for s_span in rs_spans do
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      -- Sanity check against bad instances.
      regentlib.assert(s.mapsp1 ~= s.mapsp2, "bad pointer in mapsp1 and/or mapsp2")

      s.mapsp1 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp1)
      regentlib.assert(not isnull(s.mapsp1), "dynamic_cast failed on mapsp1")
      s.mapsp2 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp2)
      regentlib.assert(not isnull(s.mapsp2), "dynamic_cast failed on mapsp2")

      -- Sanity check internal spans.
      if s_span.internal then
        var p1 = static_cast(ptr(point, rpp), s.mapsp1)
        regentlib.assert(not isnull(p1), "mapsp1 not internal")
        var p2 = static_cast(ptr(point, rpp), s.mapsp2)
        regentlib.assert(not isnull(p2), "mapsp2 not internal")
      end
    end
  end
end

task init_mesh_zones(rz : region(zone),
                     rz_spans : region(span))
where
  writes(rz.{zx, zarea, zvol}),

  reads(rz_spans)
do
  for z_span in rz_spans do
    __demand(__vectorize)
    for z_raw = z_span.start, z_span.stop do
      var z = unsafe_cast(ptr(zone, rz), z_raw)

      z.zx = vec2 {x = 0.0, y = 0.0}
      z.zarea = 0.0
      z.zvol = 0.0
    end
  end
end

task calc_centers_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       rs_spans : region(span),
                       enable : bool)
where
  reads(rz.znump, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zx, rs.ex),

  reads(rs_spans)
do
  if not enable then return end

  for s_span in rs_spans do
    var zx = vec2 { x = 0.0, y = 0.0 }
    var nside = 1
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2
      var e = s

      var p1_px = p1.px
      e.ex = 0.5*(p1_px + p2.px)

      zx += p1_px

      if nside == z.znump then
        z.zx = (1/double(z.znump)) * zx
        zx = vec2 { x = 0.0, y = 0.0 }
        nside = 0
      end
      nside += 1
    end
  end
end

task calc_volumes_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       rs_spans : region(span),
                       enable : bool)
where
  reads(rz.{zx, znump}, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zarea, zvol}, rs.{sarea}),

  reads(rs_spans)
do
  if not enable then return end

  for s_span in rs_spans do
    var zarea = 0.0
    var zvol = 0.0
    var nside = 1
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      var z = s.mapsz
      var p1 = s.mapsp1
      var p2 = s.mapsp2

      var p1_px = p1.px
      var p2_px = p2.px
      var sa = 0.5 * cross(p2_px - p1_px, z.zx - p1_px)
      var sv = sa * (p1_px.x + p2_px.x + z.zx.x)
      s.sarea = sa
      -- s.svol = sv

      zarea += sa
      zvol += sv

      if nside == z.znump then
        z.zarea = zarea
        z.zvol = (1.0 / 3.0) * zvol
        zarea = 0.0
        zvol = 0.0
        nside = 0
      end
      nside += 1

      regentlib.assert(sv > 0.0, "sv negative")
    end
  end
end

task init_side_fracs(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)),
                     rs_spans : region(span))
where
  reads(rz.zarea, rs.{mapsz, sarea}),
  writes(rs.smf),

  reads(rs_spans)
do
  for s_span in rs_spans do
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      var z = s.mapsz
      s.smf = s.sarea / z.zarea
    end
  end
end

task init_hydro(rz : region(zone),
                rz_spans : region(span),
                rinit : double, einit : double,
                rinitsub : double, einitsub : double,
                subregion_x0 : double, subregion_x1 : double,
                subregion_y0 : double, subregion_y1 : double)
where
  reads(rz.{zx, zvol}),
  writes(rz.{zr, ze, zwrate, zm, zetot}),

  reads(rz_spans)
do
  for z_span in rz_spans do
    for z_raw = z_span.start, z_span.stop do
      var z = unsafe_cast(ptr(zone, rz), z_raw)

      var zr = rinit
      var ze = einit

      var eps = 1e-12
      if z.zx.x > subregion_x0 - eps and
        z.zx.x < subregion_x1 + eps and
        z.zx.y > subregion_y0 - eps and
        z.zx.y < subregion_y1 + eps
      then
        zr = rinitsub
        ze = einitsub
      end

      var zm = zr * z.zvol

      z.zr = zr
      z.ze = ze
      z.zwrate = 0.0
      z.zm = zm
      z.zetot = ze * zm
    end
  end
end

task init_radial_velocity(rp : region(point),
                          rp_spans : region(span),
                          vel : double)
where
  reads(rp.px),
  writes(rp.pu),

  reads(rp_spans)
do
  for p_span in rp_spans do
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      if vel == 0.0 then
        p.pu = {x = 0.0, y = 0.0}
      else
        var pmag = length(p.px)
        p.pu = (vel / pmag)*p.px
      end
    end
  end
end

-- #####################################
-- ## Main simulation loop
-- #################

task adv_pos_half(rp : region(point),
                  rp_spans : region(span),
                  dt : double,
                  enable : bool,
                  print_ts : bool)
where
  writes(rp.{pmaswt, pf}),

  reads(rp.{px, pu}),
  writes(rp.{px0, pxp, pu0}),

  reads(rp_spans)
do
  if not enable then return end

  if print_ts then c.printf("t: %ld\n", c.legion_get_current_time_in_micros()) end

  var dth = 0.5 * dt

  for p_span in rp_spans do
    -- Save off point variable values from previous cycle.
    -- Initialize fields used in reductions.
    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      p.pmaswt = 0.0
    end
    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      p.pf.x = 0.0
    end
    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      p.pf.y = 0.0
    end

    --
    -- 1. Advance mesh to center of time step.
    --

    -- Copy state variables from previous time step and update position.
    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      var px0_x = p.px.x
      var pu0_x = p.pu.x
      p.px0.x = px0_x
      p.pu0.x = pu0_x
      p.pxp.x = px0_x + dth*pu0_x
    end
    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      var px0_y = p.px.y
      var pu0_y = p.pu.y
      p.px0.y = px0_y
      p.pu0.y = pu0_y
      p.pxp.y = px0_y + dth*pu0_y
    end
  end
end

task calc_everything(rz : region(zone), rpp : region(point), rpg : region(point),
                     rs : region(side(rz, rpp, rpg, rs)),
                     rz_spans : region(span),
                     rs_spans : region(span),
                     alfa : double, gamma : double, ssmin : double, dt : double,
                     q1 : double, q2 : double,
                     enable : bool)
where
  reads(rz.zvol),
  writes(rz.zvol0),

  reads(rz.znump, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zxp, rs.exp),

  reads(rz.{zxp, znump}, rpp.pxp, rpg.pxp, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zareap, zvolp}, rs.{sareap, elen}),

  reads(rz.znump, rs.{mapsz, sareap, elen}),
  writes(rz.zdl),

  reads(rz.{zvolp, zm}),
  writes(rz.zrp),

  reads(rz.{zareap, zrp}, rs.{mapsz, mapsp1, mapss3, smf}),
  reads writes(rpp.pmaswt),
  reduces+(rpg.pmaswt),

  reads(rz.{zvol0, zvolp, zm, zr, ze, zwrate}),
  writes(rz.{zp, zss}),

  reads(rz.{zxp, zareap, zrp, zss, zp}, rs.{mapsz, sareap, smf, exp}),
  writes(rs.{sfp, sft}),

  reads(rz.znump, rpp.pu, rpg.pu, rs.{mapsz, mapsp1}),
  writes(rz.zuc),

  reads(rz.{zxp, zuc}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsz, mapsp1, mapsp2, mapss3, exp, elen}),
  writes(rs.{carea, ccos, cdiv, cevol, cdu}),

  reads(rz.{zrp, zss}, rpp.pu, rpg.pu,
        rs.{mapsz, mapsp1, mapsp2, mapss3, elen, cdiv, cdu, cevol}),
  writes(rs.{cqe1, cqe2}),

  reads(rs.{mapss4, elen, carea, ccos, cqe1, cqe2}),
  writes(rs.sfq),

  reads(rz.{zss, z0tmp}, rpp.{pxp, pu}, rpg.{pxp, pu},
        rs.{mapsp1, mapsp2, mapsz, elen}),
  writes(rz.{zdu, z0tmp}),

  reads(rz.znump, rs.{mapsz, mapsp1, mapss3, sfq, sft}),
  reads writes(rpp.pf),
  reduces+(rpg.pf.{x, y}),

  reads(rs_spans, rz_spans)
do
  if not enable then return end

  for s_span in rs_spans do
    var z_span = rz_spans[unsafe_cast(ptr(span, rz_spans), s_span)]

    -- Save off zone variable value from previous cycle.
    -- Copy state variables from previous time step.
    __demand(__vectorize)
    for z_raw = z_span.start, z_span.stop do
      var z = unsafe_cast(ptr(zone, rz), z_raw)

      z.zvol0 = z.zvol
    end

    --
    -- 1a. Compute new mesh geometry.
    --

    -- Compute centers of zones and edges.
    if s_span.internal then
      var zxp = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)
        var e = s

        var p1_pxp = p1.pxp
        e.exp = vec2_mul_lhs(0.5, vec2_add(p1_pxp, p2.pxp))

        zxp += p1_pxp

        if nside == z.znump then
          z.zxp = (1/double(z.znump)) * zxp
          zxp = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    else
      var zxp = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2
        var e = s

        var p1_pxp = p1.pxp
        e.exp = vec2_mul_lhs(0.5, vec2_add(p1_pxp, p2.pxp))

        zxp += p1_pxp

        if nside == z.znump then
          z.zxp = (1/double(z.znump)) * zxp
          zxp = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    end

    -- Compute volumes of zones and sides.
    -- Compute edge lengths.
    if s_span.internal then
      var zareap = 0.0
      var zvolp = 0.0
      var nside = 1
      var numsbad = 0
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)

        var p1_pxp = p1.pxp
        var p2_pxp = p2.pxp
        var sa = 0.5 * vec2_cross(vec2_sub(p2_pxp, p1_pxp), vec2_sub(z.zxp, p1_pxp))
        var sv = sa * (p1_pxp.x + p2_pxp.x + z.zxp.x)
        s.sareap = sa
        -- s.svolp = sv
        s.elen = vec2_length(vec2_sub(p2_pxp, p1_pxp))

        zareap += sa
        zvolp += sv

        if nside == z.znump then
          z.zareap = zareap
          z.zvolp = (1.0 / 3.0) * zvolp
          zareap = 0.0
          zvolp = 0.0
          nside = 0
        end
        nside += 1

        numsbad += int(sv <= 0.0)
      end
      regentlib.assert(numsbad == 0, "sv negative")
    else
      var zareap = 0.0
      var zvolp = 0.0
      var nside = 1
      var numsbad = 0
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2

        var p1_pxp = p1.pxp
        var p2_pxp = p2.pxp
        var sa = 0.5 * vec2_cross(vec2_sub(p2_pxp, p1_pxp), vec2_sub(z.zxp, p1_pxp))
        var sv = sa * (p1_pxp.x + p2_pxp.x + z.zxp.x)
        s.sareap = sa
        -- s.svolp = sv
        s.elen = vec2_length(vec2_sub(p2_pxp, p1_pxp))

        zareap += sa
        zvolp += sv

        if nside == z.znump then
          z.zareap = zareap
          z.zvolp = (1.0 / 3.0) * zvolp
          zareap = 0.0
          zvolp = 0.0
          nside = 0
        end
        nside += 1

        numsbad += int(sv <= 0.0)
      end
      regentlib.assert(numsbad == 0, "sv negative")
    end

    -- Compute zone characteristic lengths.
    do
      var zdl = 1e99
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var e = s

        var area = s.sareap
        var base = e.elen
        var fac = 3.0 + [int](z.znump ~= 3) * 1.0
        var sdl = fac * area / base
        zdl = min(zdl, sdl)

        if nside == z.znump then
          z.zdl = zdl
          zdl = 1e99
          nside = 0
        end
        nside += 1
      end
    end

    --
    -- 2. Compute point masses.
    --

    -- Compute zone densities.
    __demand(__vectorize)
    for z_raw = z_span.start, z_span.stop do
      var z = unsafe_cast(ptr(zone, rz), z_raw)

      z.zrp = z.zm / z.zvolp
    end

    -- Reduce masses into points.
    if s_span.internal then
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var s3 = s.mapss3

        var m = z.zrp * z.zareap * 0.5 * (s.smf + s3.smf)
        p1.pmaswt += m
      end
    else
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var s3 = s.mapss3

        var m = z.zrp * z.zareap * 0.5 * (s.smf + s3.smf)
        p1.pmaswt += m
      end
    end

    --
    -- 3. Compute material state (half-advanced).
    --

    do
      var gm1 = gamma - 1.0
      var ss2 = max(ssmin * ssmin, 1e-99)
      var dth = 0.5 * dt

      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        var rx = z.zr
        var ex = max(z.ze, 0.0)
        var prex = gm1 * ex
        var perx = gm1 * rx
        var px = perx * ex
        var csqd = max(ss2, prex + gm1 * prex)
        var z0per = perx
        z.zss = sqrt(csqd)

        var zminv = 1.0 / z.zm
        var dv = (z.zvolp - z.zvol0) * zminv
        var bulk = z.zr * csqd
        var denom = 1.0 + 0.5 * z0per * dv
        var src = z.zwrate * dth * zminv
        z.zp = px + (z0per * src - z.zr * bulk * dv) / denom
      end
    end

    --
    -- 4. Compute forces.
    --

    -- Compute PolyGas and TTS forces.
    __demand(__vectorize)
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      var z = s.mapsz

      -- Compute surface vectors of sides.
      var ssurfp = s.exp
      ssurfp.x -= z.zxp.x
      ssurfp.y -= z.zxp.y
      var tmp = -ssurfp.y
      ssurfp.y = ssurfp.x
      ssurfp.x = tmp

      -- Compute PolyGas forces.
      var sfx = ssurfp
      var zp = -z.zp
      sfx.x *= zp
      sfx.y *= zp
      s.sfp = sfx

      -- Compute TTS forces.
      var svfacinv = z.zareap / s.sareap
      var srho = z.zrp * s.smf * svfacinv
      var sstmp = max(z.zss, ssmin)
      sstmp = alfa * sstmp * sstmp
      var sdp = sstmp * (srho - z.zrp)
      var sqq = vec2_mul_lhs(-sdp, ssurfp)
      s.sft = vec2_add(sfx, sqq)
    end

    -- Compute QCS forces.

    -- QCS zone center velocity.
    if s_span.internal then
      var zuc = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)

        zuc += p1.pu

        if nside == z.znump then
          z.zuc = (1.0 / double(z.znump))*zuc
          zuc = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    else
      var zuc = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1

        zuc += p1.pu

        if nside == z.znump then
          z.zuc = (1.0 / double(z.znump))*zuc
          zuc = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    end

    -- QCS corner divergence.
    if s_span.internal then
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s2 = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var c = s2
        var s = s2.mapss3
        var z = s.mapsz
        var p = unsafe_cast(ptr(point, rpp), s.mapsp2)
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s2.mapsp2)
        var e1 = s
        var e2 = s2

        -- velocities and positions
        -- point p
        var up0 = p.pu
        var xp0 = p.pxp
        -- edge e2
        var up1 = vec2_mul_lhs(0.5, vec2_add(p.pu, p2.pu))
        var xp1 = e2.exp
        -- zone center z
        var up2 = z.zuc
        var xp2 = z.zxp
        -- edge e1
        var up3 = vec2_mul_lhs(0.5, vec2_add(p1.pu, p.pu))
        var xp3 = e1.exp

        -- compute 2d cartesian volume of corner
        var xp2sub0 = vec2_sub(xp2, xp0)
        var xp3sub1 = vec2_sub(xp3, xp1)
        var cvolume = 0.5 * vec2_cross(xp2sub0, xp3sub1)
        c.carea = cvolume

        -- compute cosine angle
        var v1 = vec2_sub(xp3, xp0)
        var v2 = vec2_sub(xp1, xp0)
        var de1 = e1.elen
        var de2 = e2.elen
        var minelen = min(de1, de2)
        var cond1 = [int](minelen >= 1e-12)
        c.ccos = cond1 * 4.0 * vec2_dot(v1, v2) / (de1 * de2)

        -- compute divergence of corner
        var up2sub0 = vec2_sub(up2, up0)
        var up3sub1 = vec2_sub(up3, up1)
        var cdiv = (vec2_cross(up2sub0, xp3sub1) -
                    vec2_cross(up3sub1, xp2sub0)) / (2.0 * cvolume)
        c.cdiv = cdiv

        -- compute evolution factor
        var dxx1 = vec2_mul_lhs(0.5, vec2_add(vec2_neg(xp3sub1), xp2sub0))
        var dxx2 = vec2_mul_lhs(0.5, vec2_add(xp2sub0, xp3sub1))
        var dx1 = vec2_length(dxx1)
        var dx2 = vec2_length(dxx2)

        -- average corner-centered velocity
        var duav = vec2_mul_lhs(0.25, vec2_add(vec2_add(vec2_add(up0, up1), up2), up3))

        var tmp1 = vec2_dot(dxx1, duav)
        var test1 = fabs(tmp1 * dx2)
        var tmp2 = vec2_dot(dxx2, duav)
        var test2 = fabs(tmp2 * dx1)
        var cond2 = [int](test1 > test2)
        var num = cond2 * dx1 + (1 - cond2) * dx2
        var den = cond2 * dx2 + (1 - cond2) * dx1
        var r = num / den
        var evol = min(sqrt(4.0 * cvolume * r), 2.0 * minelen)

        -- compute delta velocity
        var dv1 = vec2_length(vec2_add(vec2_neg(up3sub1), up2sub0))
        var dv2 = vec2_length(vec2_add(up2sub0, up3sub1))
        var du = max(dv1, dv2)

        var cond3 = [int](cdiv < 0.0)
        c.cevol = cond3 * evol
        c.cdu = cond3 * du
      end
    else
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s2 = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var c = s2
        var s = s2.mapss3
        var z = s.mapsz
        var p = s.mapsp2
        var p1 = s.mapsp1
        var p2 = s2.mapsp2
        var e1 = s
        var e2 = s2

        -- velocities and positions
        -- point p
        var up0 = p.pu
        var xp0 = p.pxp
        -- edge e2
        var up1 = vec2_mul_lhs(0.5, vec2_add(p.pu, p2.pu))
        var xp1 = e2.exp
        -- zone center z
        var up2 = z.zuc
        var xp2 = z.zxp
        -- edge e1
        var up3 = vec2_mul_lhs(0.5, vec2_add(p1.pu, p.pu))
        var xp3 = e1.exp

        -- compute 2d cartesian volume of corner
        var xp2sub0 = vec2_sub(xp2, xp0)
        var xp3sub1 = vec2_sub(xp3, xp1)
        var cvolume = 0.5 * vec2_cross(xp2sub0, xp3sub1)
        c.carea = cvolume

        -- compute cosine angle
        var v1 = vec2_sub(xp3, xp0)
        var v2 = vec2_sub(xp1, xp0)
        var de1 = e1.elen
        var de2 = e2.elen
        var minelen = min(de1, de2)
        var cond1 = [int](minelen >= 1e-12)
        c.ccos = cond1 * 4.0 * vec2_dot(v1, v2) / (de1 * de2)

        -- compute divergence of corner
        var up2sub0 = vec2_sub(up2, up0)
        var up3sub1 = vec2_sub(up3, up1)
        var cdiv = (vec2_cross(up2sub0, xp3sub1) -
                    vec2_cross(up3sub1, xp2sub0)) / (2.0 * cvolume)
        c.cdiv = cdiv

        -- compute evolution factor
        var dxx1 = vec2_mul_lhs(0.5, vec2_add(vec2_neg(xp3sub1), xp2sub0))
        var dxx2 = vec2_mul_lhs(0.5, vec2_add(xp2sub0, xp3sub1))
        var dx1 = vec2_length(dxx1)
        var dx2 = vec2_length(dxx2)

        -- average corner-centered velocity
        var duav = vec2_mul_lhs(0.25, vec2_add(vec2_add(vec2_add(up0, up1), up2), up3))

        var tmp1 = vec2_dot(dxx1, duav)
        var test1 = fabs(tmp1 * dx2)
        var tmp2 = vec2_dot(dxx2, duav)
        var test2 = fabs(tmp2 * dx1)
        var cond2 = [int](test1 > test2)
        var num = cond2 * dx1 + (1 - cond2) * dx2
        var den = cond2 * dx2 + (1 - cond2) * dx1
        var r = num / den
        var evol = min(sqrt(4.0 * cvolume * r), 2.0 * minelen)

        -- compute delta velocity
        var dv1 = vec2_length(vec2_add(vec2_neg(up3sub1), up2sub0))
        var dv2 = vec2_length(vec2_add(up2sub0, up3sub1))
        var du = max(dv1, dv2)

        var cond3 = [int](cdiv < 0.0)
        c.cevol = cond3 * evol
        c.cdu = cond3 * du
      end
    end

    -- QCS QCN force.
    if s_span.internal then
      var gammap1 = gamma + 1.0

      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s4 = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var c = s4
        var z = c.mapsz

        var ztmp2 = q2 * 0.25 * gammap1 * c.cdu
        var ztmp1 = q1 * z.zss
        var zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1)
        var rmu = [int](c.cdiv <= 0.0) * (zkur * z.zrp * c.cevol)

        var s = c.mapss3
        var p = unsafe_cast(ptr(point, rpp), s.mapsp2)
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var e1 = s
        var p2 = unsafe_cast(ptr(point, rpp), s4.mapsp2)
        var e2 = s4

        c.cqe1.x = rmu / e1.elen*(p.pu.x - p1.pu.x)
        c.cqe1.y = rmu / e1.elen*(p.pu.y - p1.pu.y)
        c.cqe2.x = rmu / e2.elen*(p2.pu.x - p.pu.x)
        c.cqe2.y = rmu / e2.elen*(p2.pu.y - p.pu.y)
      end
    else
      var gammap1 = gamma + 1.0

      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s4 = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var c = s4
        var z = c.mapsz

        var ztmp2 = q2 * 0.25 * gammap1 * c.cdu
        var ztmp1 = q1 * z.zss
        var zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1)
        var rmu = [int](c.cdiv <= 0.0) * (zkur * z.zrp * c.cevol)

        var s = c.mapss3
        var p = s.mapsp2
        var p1 = s.mapsp1
        var e1 = s
        var p2 = s4.mapsp2
        var e2 = s4

        c.cqe1.x = rmu / e1.elen*(p.pu.x - p1.pu.x)
        c.cqe1.y = rmu / e1.elen*(p.pu.y - p1.pu.y)
        c.cqe2.x = rmu / e2.elen*(p2.pu.x - p.pu.x)
        c.cqe2.y = rmu / e2.elen*(p2.pu.y - p.pu.y)
      end
    end

    -- QCS force.
    __demand(__vectorize)
    for s_raw = s_span.start, s_span.stop do
      var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

      var c1 = s
      var c2 = s.mapss4
      var e = s
      var el = e.elen

      var c1sin2 = 1.0 - c1.ccos * c1.ccos
      var cond1 = [int](c1sin2 >= 1e-4)
      var c1w = cond1 * (c1.carea / c1sin2)
      var c1cos = cond1 * c1.ccos

      var c2sin2 = 1.0 - c2.ccos * c2.ccos
      var cond2 = [int](c2sin2 >= 1e-4)
      var c2w = cond2 * (c2.carea / c2sin2)
      var c2cos = cond2 * c2.ccos

      s.sfq.x = (1.0 / el)*(c1w*(c1.cqe2.x + c1cos*c1.cqe1.x) +
                            c2w*(c2.cqe1.x + c2cos*c2.cqe2.x))
      s.sfq.y = (1.0 / el)*(c1w*(c1.cqe2.y + c1cos*c1.cqe1.y) +
                            c2w*(c2.cqe1.y + c2cos*c2.cqe2.y))
    end

    -- QCS vel diff.
    if s_span.internal then
      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        z.z0tmp = 0.0
      end

      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)
        var z = s.mapsz
        var e = s

        var dx = vec2_sub(p2.pxp, p1.pxp)
        var du = vec2_sub(p2.pu, p1.pu)
        var lenx = e.elen
        var dux = vec2_dot(du, dx)
        dux = [int](lenx > 0.0) * (fabs(dux) / lenx)
        z.z0tmp max= dux
      end

      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        z.zdu = q1 * z.zss + 2.0 * q2 * z.z0tmp
      end
    else
      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        z.z0tmp = 0.0
      end

      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var p1 = s.mapsp1
        var p2 = s.mapsp2
        var z = s.mapsz
        var e = s

        var dx = vec2_sub(p2.pxp, p1.pxp)
        var du = vec2_sub(p2.pu, p1.pu)
        var lenx = e.elen
        var dux = du.x * dx.x + du.y * dx.y
        dux = [int](lenx > 0.0) * (fabs(dux) / lenx)
        z.z0tmp max= dux
      end

      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        z.zdu = q1 * z.zss + 2.0 * q2 * z.z0tmp
      end
    end

    -- Reduce forces into points.
    if s_span.internal then
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var s3 = s.mapss3

        var f = vec2_sub(vec2_add(s.sfq, s.sft), vec2_add(s3.sfq, s3.sft))
        p1.pf.x += f.x
        p1.pf.y += f.y
      end
    else
      __demand(__vectorize)
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var p1 = s.mapsp1
        var s3 = s.mapss3

        var f = vec2_sub(vec2_add(s.sfq, s.sft), vec2_add(s3.sfq, s3.sft))
        p1.pf.x += f.x
        p1.pf.y += f.y
      end
    end
  end
end

task adv_pos_full(rp : region(point),
                  rp_spans : region(span),
                  dt : double,
                  enable : bool)
where
  reads(rp.{has_bcx, has_bcy}),
  reads writes(rp.{pu0, pf}),

  reads(rp.{px0, pu0, pf, pmaswt}),
  writes(rp.{px, pu}),

  reads(rp_spans)
do
  if not enable then return end

  for p_span in rp_spans do
    --
    -- 4a. Apply boundary conditions.
    --

    __demand(__vectorize)
    for p_raw = p_span.start, p_span.stop do
      var p = unsafe_cast(ptr(point, rp), p_raw)

      var cond_x = 1 - [int](p.has_bcx)
      var cond_y = 1 - [int](p.has_bcy)
      p.pu0.x *= cond_x
      p.pf.x *= cond_x
      p.pu0.y *= cond_y
      p.pf.y *= cond_y
    end

    --
    -- 5. Compute accelerations.
    -- 6. Advance mesh to end of time step.
    --

    do
      var fuzz = 1e-99
      var dth = 0.5 * dt
      __demand(__vectorize)
      for p_raw = p_span.start, p_span.stop do
        var p = unsafe_cast(ptr(point, rp), p_raw)

        var fac = 1.0 / max(p.pmaswt, fuzz)
        var pap_x = fac*p.pf.x
        var pap_y = fac*p.pf.y

        var pu_x = p.pu0.x + dt*(pap_x)
        p.pu.x = pu_x
        p.px.x = p.px0.x + dth*(pu_x + p.pu0.x)

        var pu_y = p.pu0.y + dt*(pap_y)
        p.pu.y = pu_y
        p.px.y = p.px0.y + dth*(pu_y + p.pu0.y)
      end
    end
  end
end

task calc_everything_full(rz : region(zone), rpp : region(point), rpg : region(point),
                          rs : region(side(rz, rpp, rpg, rs)),
                          rz_spans : region(span),
                          rs_spans : region(span),
                          dt : double,
                          enable : bool)
where
  reads(rz.znump, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zx, rs.ex),

  reads(rz.{zx, znump}, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zarea, zvol}, rs.{sarea}),

  reads(rz.{zetot, znump}, rpp.{pxp, pu0, pu}, rpg.{pxp, pu0, pu},
        rs.{mapsz, mapsp1, mapsp2, sfp, sfq}),
  writes(rz.{zw, zetot}),

  reads(rz.{zvol0, zvol, zm, zw, zp, zetot}),
  writes(rz.{zwrate, ze, zr}),

  reads(rs_spans, rz_spans)
do
  if not enable then return end

  for s_span in rs_spans do
    var z_span = rz_spans[unsafe_cast(ptr(span, rz_spans), s_span)]

    --
    -- 6a. Compute new mesh geometry.
    --

    -- Calc centers.
    if s_span.internal then
      var zx = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)
        var e = s

        var p1_px = p1.px
        e.ex = vec2_mul_lhs(0.5, vec2_add(p1_px, p2.px))

        zx += p1_px

        if nside == z.znump then
          z.zx = (1/double(z.znump)) * zx
          zx = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    else
      var zx = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2
        var e = s

        var p1_px = p1.px
        e.ex = vec2_mul_lhs(0.5, vec2_add(p1_px, p2.px))

        zx += p1_px

        if nside == z.znump then
          z.zx = (1/double(z.znump)) * zx
          zx = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    end

    -- Calc volumes.
    if s_span.internal then
      var zarea = 0.0
      var zvol = 0.0
      var nside = 1
      var numsbad = 0
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)

        var p1_px = p1.px
        var p2_px = p2.px
        var sa = 0.5 * cross(p2_px - p1_px, z.zx - p1_px)
        var sv = sa * (p1_px.x + p2_px.x + z.zx.x)
        s.sarea = sa
        -- s.svol = sv

        zarea += sa
        zvol += sv

        if nside == z.znump then
          z.zarea = zarea
          z.zvol = (1.0 / 3.0) * zvol
          zarea = 0.0
          zvol = 0.0
          nside = 0
        end
        nside += 1

        numsbad += int(sv <= 0.0)
      end
      regentlib.assert(numsbad == 0, "sv negative")
    else
      var zarea = 0.0
      var zvol = 0.0
      var nside = 1
      var numsbad = 0
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2

        var p1_px = p1.px
        var p2_px = p2.px
        var sa = 0.5 * cross(p2_px - p1_px, z.zx - p1_px)
        var sv = sa * (p1_px.x + p2_px.x + z.zx.x)
        s.sarea = sa
        -- s.svol = sv

        zarea += sa
        zvol += sv

        if nside == z.znump then
          z.zarea = zarea
          z.zvol = (1.0 / 3.0) * zvol
          zarea = 0.0
          zvol = 0.0
          nside = 0
        end
        nside += 1

        numsbad += int(sv <= 0.0)
      end
      regentlib.assert(numsbad == 0, "sv negative")
    end

    --
    -- 7. Compute work
    --

    if s_span.internal then
      var zdwork = 0.0
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = unsafe_cast(ptr(point, rpp), s.mapsp1)
        var p2 = unsafe_cast(ptr(point, rpp), s.mapsp2)

        var sftot = vec2_add(s.sfp, s.sfq)
        var sd1 = vec2_dot(sftot, vec2_add(p1.pu0, p1.pu))
        var sd2 = vec2_dot(vec2_mul_lhs(-1.0, sftot), vec2_add(p2.pu0, p2.pu))
        var dwork = -0.5 * dt * (sd1 * p1.pxp.x + sd2 * p2.pxp.x)

        zdwork += dwork

        if nside == z.znump then
          z.zetot += zdwork
          z.zw = zdwork
          zdwork = 0.0
          nside = 0
        end
        nside += 1
      end
    else
      var zdwork = 0.0
      var nside = 1
      for s_raw = s_span.start, s_span.stop do
        var s = unsafe_cast(ptr(side(rz, rpp, rpg, rs), rs), s_raw)

        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2

        var sftot = vec2_add(s.sfp, s.sfq)
        var sd1 = vec2_dot(sftot, vec2_add(p1.pu0, p1.pu))
        var sd2 = vec2_dot(vec2_mul_lhs(-1.0, sftot), vec2_add(p2.pu0, p2.pu))
        var dwork = -0.5 * dt * (sd1 * p1.pxp.x + sd2 * p2.pxp.x)

        zdwork += dwork

        if nside == z.znump then
          z.zetot += zdwork
          z.zw = zdwork
          zdwork = 0.0
          nside = 0
        end
        nside += 1
      end
    end

    --
    -- 7a. Compute work rate.
    --

    do
      var dtiny = 1.0 / dt
      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        var dvol = z.zvol - z.zvol0
        z.zwrate = (z.zw + z.zp * dvol) * dtiny
      end
    end

    --
    -- 8. Update state variables.
    --

    do
      var fuzz = 1e-99
      __demand(__vectorize)
      for z_raw = z_span.start, z_span.stop do
        var z = unsafe_cast(ptr(zone, rz), z_raw)

        z.ze = z.zetot / (z.zm + fuzz)
      end
    end

    __demand(__vectorize)
    for z_raw = z_span.start, z_span.stop do
      var z = unsafe_cast(ptr(zone, rz), z_raw)

      z.zr = z.zm / z.zvol
    end
  end
end

--
-- 9. Compute timstep for next cycle.
--

task calc_dt_hydro(rz : region(zone),
                   rz_spans : region(span),
                   dtlast : double, dtmax : double,
                   cfl : double, cflv : double,
                   enable : bool,
                   print_ts : bool) : double
where
  reads(rz.{zdl, zvol0, zvol, zss, zdu}),

  reads(rz_spans)
do
  var dthydro = dtmax

  if not enable then return dthydro end

  -- Calc dt courant.
  do
    var fuzz = 1e-99
    var dtnew = dtmax
    --__demand(__vectorize)
    for z in rz do
      var cdu = max(z.zdu, max(z.zss, fuzz))
      var zdthyd = z.zdl * cfl / cdu

      dtnew min= zdthyd
    end

    dthydro min= dtnew
  end

  -- Calc dt volume.
  do
    var dvovmax = 1e-99
    --__demand(__vectorize)
    for z in rz do
      var zdvov = abs((z.zvol - z.zvol0) / z.zvol0)
      dvovmax max= zdvov
    end
    dthydro min= dtlast * cflv / dvovmax
  end

  if print_ts then c.printf("t: %ld\n", c.legion_get_current_time_in_micros()) end

  return dthydro
end

task calc_global_dt(dt : double, dtfac : double, dtinit : double,
                    dtmax : double, dthydro : double,
                    time : double, tstop : double, cycle : int64) : double
  var dtlast = dt

  dt = dtmax

  if cycle == 0 then
    dt = min(dt, dtinit)
  else
    var dtrecover = dtfac * dtlast
    dt = min(dt, dtrecover)
  end

  dt = min(dt, tstop - time)
  dt = min(dt, dthydro)

  return dt
end

terra wait_for(x : double)
  return x
end

task continue_simulation(warmup : bool,
                         cycle : int64, cstop : int64,
                         time : double, tstop : double)
  return warmup or (cycle < cstop and time < tstop)
end

task read_input_sequential(rz_all : region(zone),
                           rp_all : region(point),
                           rs_all : region(side(wild, wild, wild, wild)),
                           conf : config)
where reads writes(rz_all, rp_all, rs_all) do
  return read_input(
    __runtime(), __context(),
    __physical(rz_all), __fields(rz_all),
    __physical(rp_all), __fields(rp_all),
    __physical(rs_all), __fields(rs_all),
    conf)
end

terra get_raw_span(runtime : c.legion_runtime_t, ctx : c.legion_context_t,
                   r : c.legion_logical_region_t)
  var it = c.legion_index_iterator_create(runtime, ctx, r.index_space)
  while c.legion_index_iterator_has_next(it) do
    var count : c.size_t = 0
    var start = c.legion_index_iterator_next_span(it, &count, -1)
    regentlib.assert(not c.legion_index_iterator_has_next(it), "multiple spans")

    return span { start = start.value, stop = start.value + count, internal = false }
  end
  return span { start = 0, stop = 0, internal = false }
end

task read_spans_sequential(
  rz_all : region(zone),
  rz_all_p : partition(disjoint, rz_all),
  rz_spans_p : partition(disjoint, rz_all),
  rz_spans_x : cross_product(rz_all_p, rz_spans_p),
  rp_all : region(point),
  rp_all_private : region(point),
  rp_all_private_p : partition(disjoint, rp_all_private),
  rp_all_ghost : region(point),
  rp_all_shared_p : partition(disjoint, rp_all_ghost),
  rp_spans_p : partition(disjoint, rp_all),
  rp_spans_private_x : cross_product(rp_all_private_p, rp_spans_p),
  rp_spans_shared_x : cross_product(rp_all_shared_p, rp_spans_p),
  rs_all : region(side(wild, wild, wild, wild)),
  rs_all_p : partition(disjoint, rs_all),
  rs_spans_p : partition(disjoint, rs_all),
  rs_spans_x : cross_product(rs_all_p, rs_spans_p),
  rz_spans : region(span),
  rp_spans_private : region(span),
  rp_spans_shared : region(span),
  rs_spans : region(span),
  conf : config)
where
  reads writes(rz_spans, rp_spans_private, rp_spans_shared, rs_spans),
  rp_all_private * rp_all_ghost,
  rz_spans * rp_spans_private, rz_spans * rp_spans_shared, rz_spans * rs_spans,
  rp_spans_private * rp_spans_shared, rp_spans_private * rs_spans,
  rp_spans_shared * rs_spans
do
  if true then -- Hack: Avoid analyzing this.
  for i = 0, conf.npieces do
    for j = 0, conf.nspans_zones do
      var z = unsafe_cast(ptr(span, rz_spans), i*conf.nspans_zones + j)
      @z = get_raw_span(__runtime(), __context(), __raw(rz_spans_x[i][j]))
    end
    for j = 0, conf.nspans_zones do
      var s = unsafe_cast(ptr(span, rs_spans), i*conf.nspans_zones + j)
      @s = get_raw_span(__runtime(), __context(), __raw(rs_spans_x[i][j]))
    end
    for j = 0, conf.nspans_points do
      var p = unsafe_cast(ptr(span, rp_spans_private), i*conf.nspans_points + j)
      @p = get_raw_span(__runtime(), __context(), __raw(rp_spans_private_x[i][j]))
    end
    for j = 0, conf.nspans_points do
      var p = unsafe_cast(ptr(span, rp_spans_shared), i*conf.nspans_points + j)
      @p = get_raw_span(__runtime(), __context(), __raw(rp_spans_shared_x[i][j]))
    end
  end
  end
end

task validate_output_sequential(rz_all : region(zone),
                                rp_all : region(point),
                                rs_all : region(side(wild, wild, wild, wild)),
                                conf : config)
where reads(rz_all, rp_all, rs_all) do
  validate_output(
    __runtime(), __context(),
    __physical(rz_all), __fields(rz_all),
    __physical(rp_all), __fields(rp_all),
    __physical(rs_all), __fields(rs_all),
    conf)
end

task dummy(rz : region(zone), rpp : region(point)) : int
where reads writes(rz, rpp) do
  return 1
end

terra unwrap(x : mesh_colorings) return x end

task test()
  c.printf("Running test (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)

  var conf : config = read_config()

  var rz_all = region(ispace(ptr, conf.nz), zone)
  var rp_all = region(ispace(ptr, conf.np), point)
  var rs_all = region(ispace(ptr, conf.ns), side(wild, wild, wild, wild))

  c.printf("Reading input (t=%.1f)...\n", c.legion_get_current_time_in_micros()/1.e6)

  var colorings : mesh_colorings

  regentlib.assert(conf.seq_init or conf.par_init,
                   "enable one of sequential or parallel initialization")

  if conf.seq_init then
    -- Hack: This had better run on the same node...
    colorings = unwrap(read_input_sequential(
      rz_all, rp_all, rs_all, conf))
  end

  if conf.par_init then
    var colorings_ = read_partitions(conf)
    if conf.seq_init then
      regentlib.assert(colorings.nspans_zones == colorings_.nspans_zones, "bad nspans zones")
      regentlib.assert(colorings.nspans_points == colorings_.nspans_points, "bad nspans points")
      c.legion_coloring_destroy(colorings.rz_all_c)
      c.legion_coloring_destroy(colorings.rz_spans_c)
      c.legion_coloring_destroy(colorings.rp_all_c)
      c.legion_coloring_destroy(colorings.rp_all_private_c)
      c.legion_coloring_destroy(colorings.rp_all_ghost_c)
      c.legion_coloring_destroy(colorings.rp_all_shared_c)
      c.legion_coloring_destroy(colorings.rp_spans_c)
      c.legion_coloring_destroy(colorings.rs_all_c)
      c.legion_coloring_destroy(colorings.rs_spans_c)
    end
    colorings = colorings_
  end

  conf.nspans_zones = colorings.nspans_zones
  conf.nspans_points = colorings.nspans_points

  -- Partition zones into disjoint pieces.
  var rz_all_p = partition(disjoint, rz_all, colorings.rz_all_c)

  -- Partition points into private and ghost regions.
  var rp_all_p = partition(disjoint, rp_all, colorings.rp_all_c)
  var rp_all_private = rp_all_p[0]
  var rp_all_ghost = rp_all_p[1]

  -- Partition private points into disjoint pieces by zone.
  var rp_all_private_p = partition(
    disjoint, rp_all_private, colorings.rp_all_private_c)

  -- Partition ghost points into aliased pieces by zone.
  var rp_all_ghost_p = partition(
    aliased, rp_all_ghost, colorings.rp_all_ghost_c)

  -- Partition ghost points into disjoint pieces, breaking ties
  -- between zones so that each point goes into one region only.
  var rp_all_shared_p = partition(
    disjoint, rp_all_ghost, colorings.rp_all_shared_c)

  -- Partition sides into disjoint pieces by zone.
  var rs_all_p = partition(disjoint, rs_all, colorings.rs_all_c)

  -- Create regions and partitions for spans.
  var rz_spans = region(ispace(ptr, conf.npieces * conf.nspans_zones), span)
  var rz_spans_p = partition(equal, rz_spans, ispace(int1d, conf.npieces))

  var rp_spans_private = region(ispace(ptr, conf.npieces * conf.nspans_points), span)
  var rp_spans_private_p = partition(equal, rp_spans_private, ispace(int1d, conf.npieces))

  var rp_spans_shared = region(ispace(ptr, conf.npieces * conf.nspans_points), span)
  var rp_spans_shared_p = partition(equal, rp_spans_shared, ispace(int1d, conf.npieces))

  var rs_spans = region(ispace(ptr, conf.npieces * conf.nspans_zones), span)
  var rs_spans_p = partition(equal, rs_spans, ispace(int1d, conf.npieces))

  fill(rz_spans.{start, stop}, 0)
  fill(rp_spans_private.{start, stop}, 0)
  fill(rp_spans_shared.{start, stop}, 0)
  fill(rs_spans.{start, stop}, 0)

  -- Handle sequential span initialization.
  if conf.seq_init then
    var rz_spans_p = partition(disjoint, rz_all, colorings.rz_spans_c)
    var rz_spans_x = cross_product(rz_all_p, rz_spans_p)

    var rp_spans_p = partition(
      disjoint, rp_all, colorings.rp_spans_c)
    var rp_spans_private_x = cross_product(rp_all_private_p, rp_spans_p)
    var rp_spans_shared_x = cross_product(rp_all_shared_p, rp_spans_p)

    var rs_spans_p = partition(
      disjoint, rs_all, colorings.rs_spans_c)
    var rs_spans_x = cross_product(rs_all_p, rs_spans_p)

    read_spans_sequential(
      rz_all, rz_all_p, rz_spans_p, rz_spans_x,
      rp_all, rp_all_private, rp_all_private_p, rp_all_ghost, rp_all_shared_p,
      rp_spans_p, rp_spans_private_x, rp_spans_shared_x,
      rs_all, rs_all_p, rs_spans_p, rs_spans_x,
      rz_spans, rp_spans_private, rp_spans_shared, rs_spans,
      conf)
  end

  var par_init = [int64](conf.par_init)

  var einit = conf.einit
  var einitsub = conf.einitsub
  var rinit = conf.rinit
  var rinitsub = conf.rinitsub
  var subregion = conf.subregion

  var npieces = conf.npieces

  var subregion_0, subregion_1, subregion_2, subregion_3 = conf.subregion[0], conf.subregion[1], conf.subregion[2], conf.subregion[3]

  var prune = conf.prune

  var alfa = conf.alfa
  var cfl = conf.cfl
  var cflv = conf.cflv
  var cstop = conf.cstop + 2*prune
  var dtfac = conf.dtfac
  var dtinit = conf.dtinit
  var dtmax = conf.dtmax
  var gamma = conf.gamma
  var q1 = conf.q1
  var q2 = conf.q2
  var qgamma = conf.qgamma
  var ssmin = conf.ssmin
  var tstop = conf.tstop
  var uinitradial = conf.uinitradial
  var vfix = {x = 0.0, y = 0.0}

  var enable = true -- conf.enable and not conf.warmup
  var warmup = false -- conf.warmup and conf.enable
  var requested_print_ts = conf.print_ts
  var print_ts = requested_print_ts

  var interval = 100
  var start_time = c.legion_get_current_time_in_micros()/1.e6
  var last_time = start_time

  var time = 0.0
  var cycle : int64 = 0
  var dt = dtmax
  var dthydro = dtmax

  __demand(__spmd)
  do
    -- Initialization
    for _ = 0, par_init do
      -- __demand(__parallel)
      for i = 0, npieces do
        initialize_topology(conf, i, rz_all_p[i],
                            rp_all_private_p[i],
                            rp_all_shared_p[i],
                            rp_all_ghost_p[i],
                            rs_all_p[i])
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        initialize_spans(
          conf, i,
          rz_spans_p[i], rp_spans_private_p[i], rp_spans_shared_p[i], rs_spans_p[i])
      end
    end

    for i = 0, npieces do
      init_pointers(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans_p[i])
    end

    for i = 0, npieces do
      init_mesh_zones(
        rz_all_p[i], rz_spans_p[i])
    end

    for i = 0, npieces do
      calc_centers_full(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans_p[i],
        true)
    end

    for i = 0, npieces do
      calc_volumes_full(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans_p[i],
        true)
    end

    for i = 0, npieces do
      init_side_fracs(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans_p[i])
    end

    for i = 0, npieces do
      init_hydro(
        rz_all_p[i], rz_spans_p[i],
        rinit, einit, rinitsub, einitsub,
        subregion_0, subregion_1, subregion_2, subregion_3)
    end

    for i = 0, npieces do
      init_radial_velocity(
        rp_all_private_p[i], rp_spans_private_p[i],
        uinitradial)
    end
    for i = 0, npieces do
      init_radial_velocity(
        rp_all_shared_p[i], rp_spans_shared_p[i],
        uinitradial)
    end
  end

  __demand(__spmd)
  do
    -- Main Simulation Loop
    __demand(__trace)
    while continue_simulation(warmup, cycle, cstop, time, tstop) do
      -- if warmup and cycle > 0 then
      --   wait_for(dthydro)
      --   enable = true
      --   warmup = false
      --   time = 0.0
      --   cycle = 0
      --   dt = dtmax
      --   dthydro = dtmax
      --   start_time = c.legion_get_current_time_in_micros()/1.e6
      --   last_time = start_time
      -- end

      -- c.legion_runtime_begin_trace(__runtime(), __context(), 0, false)

      dt = calc_global_dt(dt, dtfac, dtinit, dtmax, dthydro, time, tstop, cycle)

      -- if cycle > 0 and cycle % interval == 0 then
      --   var current_time = c.legion_get_current_time_in_micros()/1.e6
      --   c.printf("cycle %4ld    sim time %.3e    dt %.3e    time %.3e (per iteration) %.3e (total)\n",
      --            cycle, time, dt, (current_time - last_time)/interval, current_time - start_time)
      --   last_time = current_time
      -- end

      print_ts = requested_print_ts and cycle == prune

      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_half(rp_all_private_p[i],
                     rp_spans_private_p[i],
                     dt,
                     enable, print_ts)
      end
      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_half(rp_all_shared_p[i],
                     rp_spans_shared_p[i],
                     dt,
                     enable, print_ts)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        calc_everything(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        rz_spans_p[i],
                        rs_spans_p[i],
                        alfa, gamma, ssmin, dt,
                        q1, q2,
                        enable)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_full(rp_all_private_p[i],
                     rp_spans_private_p[i],
                     dt,
                     enable)
      end
      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_full(rp_all_shared_p[i],
                     rp_spans_shared_p[i],
                     dt,
                     enable)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        calc_everything_full(rz_all_p[i],
                             rp_all_private_p[i],
                             rp_all_ghost_p[i],
                             rs_all_p[i],
                             rz_spans_p[i],
                             rs_spans_p[i],
                             dt,
                             enable)
      end

      print_ts = requested_print_ts and cycle == cstop - 1 - prune

      dthydro = dtmax
      -- __demand(__parallel)
      for i = 0, npieces do
        dthydro min= calc_dt_hydro(rz_all_p[i],
                                   rz_spans_p[i],
                                   dt, dtmax, cfl, cflv,
                                   enable, print_ts)
      end

      cycle += 1
      time += dt

      -- c.legion_runtime_end_trace(__runtime(), __context(), 0)
    end
  end

  if conf.seq_init then
    validate_output_sequential(
      rz_all, rp_all, rs_all, conf)
  else
    c.printf("Warning: Skipping sequential validation\n")
  end

  -- write_output(conf, rz_all, rp_all, rs_all)
end

task toplevel()
  test()
end
if os.getenv('SAVEOBJ') == '1' then
  local root_dir = arg[0]:match(".*/") or "./"
  local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
  local link_flags = terralib.newlist({"-L" .. out_dir, "-lpennant", "-lm"})

  if os.getenv('STANDALONE') == '1' then
    os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/libregent.so ' .. out_dir)
  end

  local exe = os.getenv('OBJNAME') or "pennant"
  regentlib.saveobj(toplevel, exe, "executable", cpennant.register_mappers, link_flags)
else
  regentlib.start(toplevel, cpennant.register_mappers)
end
