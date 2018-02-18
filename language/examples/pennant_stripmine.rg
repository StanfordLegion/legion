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
--    "-fflow-spmd", "1"],
--   ["pennant.tests/sedov/sedov.pnt",
--    "-npieces", "3", "-ll:cpu", "3", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-absolute", "2e-6", "-relative", "1e-8", "-relative_absolute", "1e-10",
--    "-fflow-spmd", "1"],
--   ["pennant.tests/leblanc/leblanc.pnt",
--    "-npieces", "2", "-ll:cpu", "2", "-seq_init", "1", "-par_init", "1", "-interior", "0",
--    "-fflow-spmd", "1"]
-- ]

-- Inspired by https://github.com/losalamos/PENNANT

import "regent"

require("pennant_common")

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
                   rs_spans_p : partition(disjoint, rs),
                   nspans_zones : int64)
where
  reads writes(rs.{mapsp1, mapsp2})
do
  for span = 0, nspans_zones do
    var rs_span = rs_spans_p[span]
    for s in rs_span do
      -- Sanity check against bad instances.
      regentlib.assert(s.mapsp1 ~= s.mapsp2, "bad pointer in mapsp1 and/or mapsp2")

      s.mapsp1 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp1)
      regentlib.assert(not isnull(s.mapsp1), "dynamic_cast failed on mapsp1")
      s.mapsp2 = dynamic_cast(ptr(point, rpp, rpg), s.mapsp2)
      regentlib.assert(not isnull(s.mapsp2), "dynamic_cast failed on mapsp2")
    end
  end
end

task init_mesh_zones(rz : region(zone),
                     rz_spans_p : partition(disjoint, rz),
                     nspans_zones : int64)
where
  writes(rz.{zx, zarea, zvol})
do
  for span = 0, nspans_zones do
    var rz_span = rz_spans_p[span]

    for z in rz_span do
      z.zx = vec2 {x = 0.0, y = 0.0}
      z.zarea = 0.0
      z.zvol = 0.0
    end
  end
end

task calc_centers_full(rz : region(zone), rpp : region(point), rpg : region(point),
                       rs : region(side(rz, rpp, rpg, rs)),
                       rs_spans_p : partition(disjoint, rs),
                       nspans_zones : int64,
                       enable : bool)
where
  reads(rz.znump, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.zx, rs.ex)
do
  if not enable then return end

  for span = 0, nspans_zones do
    var rs_span = rs_spans_p[span]

    var zx = vec2 { x = 0.0, y = 0.0 }
    var nside = 1
    for s in rs_span do
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
                       rs_spans_p : partition(disjoint, rs),
                       nspans_zones : int64,
                       enable : bool)
where
  reads(rz.{zx, znump}, rpp.px, rpg.px, rs.{mapsz, mapsp1, mapsp2}),
  writes(rz.{zarea, zvol}, rs.{sarea})
do
  if not enable then return end

  for span = 0, nspans_zones do
    var rs_span = rs_spans_p[span]

    var zarea = 0.0
    var zvol = 0.0
    var nside = 1
    for s in rs_span do
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
                     rs_spans_p : partition(disjoint, rs),
                     nspans_zones : int64)
where
  reads(rz.zarea, rs.{mapsz, sarea}),
  writes(rs.smf)
do
  for span = 0, nspans_zones do
    var rs_span = rs_spans_p[span]
    for s in rs_span do
      var z = s.mapsz
      s.smf = s.sarea / z.zarea
    end
  end
end

task init_hydro(rz : region(zone),
                rz_spans_p : partition(disjoint, rz),
                rinit : double, einit : double,
                rinitsub : double, einitsub : double,
                subregion_x0 : double, subregion_x1 : double,
                subregion_y0 : double, subregion_y1 : double,
                nspans_zones : int64)
where
  reads(rz.{zx, zvol}),
  writes(rz.{zr, ze, zwrate, zm, zetot})
do
  for span = 0, nspans_zones do
    var rz_span = rz_spans_p[span]
    for z in rz_span do
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
                          rp_spans_p : partition(disjoint, rp),
                          vel : double,
                          nspans_points : int64)
where
  reads(rp.px),
  writes(rp.pu)
do
  for span = 0, nspans_points do
    var rp_span = rp_spans_p[span]
    for p in rp_span do
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
                  rp_spans_p : partition(disjoint, rp),
                  dt : double,
                  nspans_points : int64,
                  enable : bool,
                  print_ts : bool)
where
  writes(rp.{pmaswt, pf}),

  reads(rp.{px, pu}),
  writes(rp.{px0, pxp, pu0})
do
  if not enable then return end

  if print_ts then c.printf("t: %ld\n", c.legion_get_current_time_in_micros()) end

  var dth = 0.5 * dt

  for span = 0, nspans_points do
    var rp_span = rp_spans_p[span]

    -- Save off point variable values from previous cycle.
    -- Initialize fields used in reductions.
    __demand(__vectorize)
    for p in rp_span do
      p.pmaswt = 0.0
    end
    __demand(__vectorize)
    for p in rp_span do
      p.pf.x = 0.0
    end
    __demand(__vectorize)
    for p in rp_span do
      p.pf.y = 0.0
    end

    --
    -- 1. Advance mesh to center of time step.
    --

    -- Copy state variables from previous time step and update position.
    __demand(__vectorize)
    for p in rp_span do
      var px0_x = p.px.x
      var pu0_x = p.pu.x
      p.px0.x = px0_x
      p.pu0.x = pu0_x
      p.pxp.x = px0_x + dth*pu0_x
    end
    __demand(__vectorize)
    for p in rp_span do
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
                     rz_spans_p : partition(disjoint, rz),
                     rs_spans_p : partition(disjoint, rs),
                     alfa : double, gamma : double, ssmin : double, dt : double,
                     q1 : double, q2 : double,
                     nspans_zones : int64,
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
  reduces+(rpg.pf.{x, y})
do
  if not enable then return end

  for span = 0, nspans_zones do
    var rz_span = rz_spans_p[span]
    var rs_span = rs_spans_p[span]

    -- Save off zone variable value from previous cycle.
    -- Copy state variables from previous time step.
    __demand(__vectorize)
    for z in rz_span do
      z.zvol0 = z.zvol
    end

    --
    -- 1a. Compute new mesh geometry.
    --

    -- Compute centers of zones and edges.
    do
      var zxp = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s in rs_span do
        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2
        var e = s

        var p1_pxp = p1.pxp
        e.exp = 0.5*(p1_pxp + p2.pxp)

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
    do
      var zareap = 0.0
      var zvolp = 0.0
      var nside = 1
      for s in rs_span do
        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2

        var p1_pxp = p1.pxp
        var p2_pxp = p2.pxp
        var sa = 0.5 * cross(p2_pxp - p1_pxp, z.zxp - p1_pxp)
        var sv = sa * (p1_pxp.x + p2_pxp.x + z.zxp.x)
        s.sareap = sa
        -- s.svolp = sv
        s.elen = length(p2_pxp - p1_pxp)

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

        regentlib.assert(sv > 0.0, "sv negative")
      end
    end

    -- Compute zone characteristic lengths.
    do
      var zdl = 1e99
      var nside = 1
      for s in rs_span do
        var z = s.mapsz
        var e = s

        var area = s.sareap
        var base = e.elen
        var fac = 0.0
        if z.znump == 3 then
          fac = 3.0
        else
          fac = 4.0
        end
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
    for z in rz_span do
      z.zrp = z.zm / z.zvolp
    end

    -- Reduce masses into points.
    for s in rs_span do
      var z = s.mapsz
      var p1 = s.mapsp1
      var s3 = s.mapss3

      var m = z.zrp * z.zareap * 0.5 * (s.smf + s3.smf)
      p1.pmaswt += m
    end

    --
    -- 3. Compute material state (half-advanced).
    --

    do
      var gm1 = gamma - 1.0
      var ss2 = max(ssmin * ssmin, 1e-99)
      var dth = 0.5 * dt

      for z in rz_span do
        var rx = z.zr
        var ex = max(z.ze, 0.0)
        var px = gm1 * rx * ex
        var prex = gm1 * ex
        var perx = gm1 * rx
        var csqd = max(ss2, prex + perx * px / (rx * rx))
        var z0per = perx
        var zss = sqrt(csqd)
        z.zss = zss

        var zminv = 1.0 / z.zm
        var dv = (z.zvolp - z.zvol0) * zminv
        var bulk = z.zr * zss * zss
        var denom = 1.0 + 0.5 * z0per * dv
        var src = z.zwrate * dth * zminv
        z.zp = px + (z0per * src - z.zr * bulk * dv) / denom
      end
    end

    --
    -- 4. Compute forces.
    --

    -- Compute PolyGas and TTS forces.
    for s in rs_span do
      var z = s.mapsz

      -- Compute surface vectors of sides.
      var ssurfp = rotateCCW(s.exp - z.zxp)

      -- Compute PolyGas forces.
      var sfx = (-z.zp)*ssurfp
      s.sfp = sfx

      -- Compute TTS forces.
      var svfacinv = z.zareap / s.sareap
      var srho = z.zrp * s.smf * svfacinv
      var sstmp = max(z.zss, ssmin)
      sstmp = alfa * sstmp * sstmp
      var sdp = sstmp * (srho - z.zrp)
      var sqq = (-sdp)*ssurfp
      s.sft = sfx + sqq
    end

    -- Compute QCS forces.

    -- QCS zone center velocity.
    do
      var zuc = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s in rs_span do
        var z = s.mapsz
        var p1 = s.mapsp1

        zuc += (1.0 / double(z.znump))*p1.pu

        if nside == z.znump then
          z.zuc = zuc
          zuc = vec2 { x = 0.0, y = 0.0 }
          nside = 0
        end
        nside += 1
      end
    end

    -- QCS corner divergence.
    for s2 in rs_span do
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
      var up1 = 0.5*(p.pu + p2.pu)
      var xp1 = e2.exp
      -- zone center z
      var up2 = z.zuc
      var xp2 = z.zxp
      -- edge e1
      var up3 = 0.5*(p1.pu + p.pu)
      var xp3 = e1.exp

      -- compute 2d cartesian volume of corner
      var cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1)
      c.carea = cvolume

      -- compute cosine angle
      var v1 = xp3 - xp0
      var v2 = xp1 - xp0
      var de1 = e1.elen
      var de2 = e2.elen
      var minelen = min(de1, de2)
      if minelen < 1e-12 then
        c.ccos = 0.0
      else
        c.ccos = 4.0 * dot(v1, v2) / (de1 * de2)
      end

      -- compute divergence of corner
      var cdiv = (cross(up2 - up0, xp3 - xp1) -
                  cross(up3 - up1, xp2 - xp0)) / (2.0 * cvolume)
      c.cdiv = cdiv

      -- compute evolution factor
      var dxx1 = 0.5*(((xp1 + xp2) - xp0) - xp3)
      var dxx2 = 0.5*(((xp2 + xp3) - xp0) - xp1)
      var dx1 = length(dxx1)
      var dx2 = length(dxx2)

      -- average corner-centered velocity
      var duav = 0.25*(((up0 + up1) + up2) + up3)

      var test1 = abs(dot(dxx1, duav) * dx2)
      var test2 = abs(dot(dxx2, duav) * dx1)
      var num = 0.0
      var den = 0.0
      if test1 > test2 then
        num = dx1
        den = dx2
      else
        num = dx2
        den = dx1
      end
      var r = num / den
      var evol = min(sqrt(4.0 * cvolume * r), 2.0 * minelen)

      -- compute delta velocity
      var dv1 = length(((up1 + up2) - up0) - up3)
      var dv2 = length(((up2 + up3) - up0) - up1)
      var du = max(dv1, dv2)

      if cdiv < 0.0 then
        c.cevol = evol
        c.cdu = du
      else
        c.cevol = 0.0
        c.cdu = 0.0
      end
    end

    -- QCS QCN force.
    do
      var gammap1 = gamma + 1.0

      for s4 in rs_span do
        var c = s4
        var z = c.mapsz

        var ztmp2 = q2 * 0.25 * gammap1 * c.cdu
        var ztmp1 = q1 * z.zss
        var zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1)
        var rmu = zkur * z.zrp * c.cevol
        if c.cdiv > 0.0 then
          rmu = 0.0
        end

        var s = c.mapss3
        var p = s.mapsp2
        var p1 = s.mapsp1
        var e1 = s
        var p2 = s4.mapsp2
        var e2 = s4

        c.cqe1 = rmu / e1.elen*(p.pu - p1.pu)
        c.cqe2 = rmu / e2.elen*(p2.pu - p.pu)
      end
    end

    -- QCS force.
    for s in rs_span do
      var c1 = s
      var c2 = s.mapss4
      var e = s
      var el = e.elen

      var c1sin2 = 1.0 - c1.ccos * c1.ccos
      var c1w = 0.0
      var c1cos = 0.0
      if c1sin2 >= 1e-4 then
        c1w = c1.carea / c1sin2
        c1cos = c1.ccos
      end

      var c2sin2 = 1.0 - c2.ccos * c2.ccos
      var c2w = 0.0
      var c2cos = 0.0
      if c2sin2 >= 1e-4 then
        c2w = c2.carea / c2sin2
        c2cos = c2.ccos
      end

      s.sfq = (1.0 / el)*(c1w*(c1.cqe2 + c1cos*c1.cqe1) +
                            c2w*(c2.cqe1 + c2cos*c2.cqe2))
    end

    -- QCS vel diff.
    do
      for z in rz_span do
        z.z0tmp = 0.0
      end

      for s in rs_span do
        var p1 = s.mapsp1
        var p2 = s.mapsp2
        var z = s.mapsz
        var e = s

        var dx = p2.pxp - p1.pxp
        var du = p2.pu - p1.pu
        var lenx = e.elen
        var dux = dot(du, dx)
        if lenx > 0.0 then
          dux = abs(dux) / lenx
        else
          dux = 0.0
        end
        z.z0tmp = max(z.z0tmp, dux)
      end

      for z in rz_span do
        z.zdu = q1 * z.zss + 2.0 * q2 * z.z0tmp
      end
    end

    -- Reduce forces into points.
    for s in rs_span do
      var p1 = s.mapsp1
      var s3 = s.mapss3

      var f = (s.sfq + s.sft) - (s3.sfq + s3.sft)
      p1.pf.x += f.x
      p1.pf.y += f.y
    end
  end
end

task adv_pos_full(rp : region(point),
                  rp_spans_p : partition(disjoint, rp),
                  dt : double,
                  nspans_points : int64,
                  enable : bool)
where
  reads(rp.{has_bcx, has_bcy}),
  reads writes(rp.{pu0, pf}),

  reads(rp.{px0, pu0, pf, pmaswt}),
  writes(rp.{px, pu})
do
  if not enable then return end

  for span = 0, nspans_points do
    var rp_span = rp_spans_p[span]

    --
    -- 4a. Apply boundary conditions.
    --

    do
      var vfixx = {x = 1.0, y = 0.0}
      var vfixy = {x = 0.0, y = 1.0}
      for p in rp_span do
        if p.has_bcx then
          p.pu0 = project(p.pu0, vfixx)
          p.pf = project(p.pf, vfixx)
        end
        if p.has_bcy then
          p.pu0 = project(p.pu0, vfixy)
          p.pf = project(p.pf, vfixy)
        end
      end
    end

    --
    -- 5. Compute accelerations.
    -- 6. Advance mesh to end of time step.
    --

    do
      var fuzz = 1e-99
      var dth = 0.5 * dt
      __demand(__vectorize)
      for p in rp_span do
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
                          rz_spans_p : partition(disjoint, rz),
                          rs_spans_p : partition(disjoint, rs),
                          dt : double,
                          nspans_zones : int64,
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
  writes(rz.{zwrate, ze, zr})
do
  if not enable then return end

  for span = 0, nspans_zones do
    var rz_span = rz_spans_p[span]
    var rs_span = rs_spans_p[span]

    --
    -- 6a. Compute new mesh geometry.
    --

    -- Calc centers.
    do
      var zx = vec2 { x = 0.0, y = 0.0 }
      var nside = 1
      for s in rs_span do
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

    -- Calc volumes.
    do
      var zarea = 0.0
      var zvol = 0.0
      var nside = 1
      for s in rs_span do
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

    --
    -- 7. Compute work
    --

    do
      var zdwork = 0.0
      var nside = 1
      for s in rs_span do
        var z = s.mapsz
        var p1 = s.mapsp1
        var p2 = s.mapsp2

        var sftot = s.sfp + s.sfq
        var sd1 = dot(sftot, p1.pu0 + p1.pu)
        var sd2 = dot(-1.0*sftot, p2.pu0 + p2.pu)
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
      for z in rz_span do
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
      for z in rz_span do
        z.ze = z.zetot / (z.zm + fuzz)
      end
    end

    __demand(__vectorize)
    for z in rz_span do
      z.zr = z.zm / z.zvol
    end
  end
end

--
-- 9. Compute timstep for next cycle.
--

task calc_dt_hydro(rz : region(zone),
                   rz_spans_p : partition(disjoint, rz),
                   dtlast : double, dtmax : double,
                   cfl : double, cflv : double,
                   nspans_zones : int64,
                   enable : bool,
                   print_ts : bool) : double
where
  reads(rz.{zdl, zvol0, zvol, zss, zdu})
do
  var dthydro = dtmax

  if not enable then return dthydro end

  -- Calc dt courant.
  do
    var fuzz = 1e-99
    var dtnew = dtmax
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

  -- Partition zones into spans.
  var rz_spans_p = partition(disjoint, rz_all, colorings.rz_spans_c)
  var rz_spans = cross_product(rz_all_p, rz_spans_p)

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

  -- Partition points into spans.
  var rp_spans_p = partition(
    disjoint, rp_all, colorings.rp_spans_c)
  var rp_spans_private = cross_product(rp_all_private_p, rp_spans_p)
  var rp_spans_shared = cross_product(rp_all_shared_p, rp_spans_p)

  -- Partition sides into disjoint pieces by zone.
  var rs_all_p = partition(disjoint, rs_all, colorings.rs_all_c)

  -- Partition sides into spans.
  var rs_spans_p = partition(
    disjoint, rs_all, colorings.rs_spans_c)
  var rs_spans = cross_product(rs_all_p, rs_spans_p)

  var par_init = [int64](conf.par_init)

  var einit = conf.einit
  var einitsub = conf.einitsub
  var rinit = conf.rinit
  var rinitsub = conf.rinitsub
  var subregion = conf.subregion

  var npieces = conf.npieces
  var nspans_zones = conf.nspans_zones
  var nspans_points = conf.nspans_points

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

  var enable = conf.enable and not conf.warmup
  var warmup = conf.warmup and conf.enable
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
    end

    for i = 0, npieces do
      init_pointers(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans[i], nspans_zones)
    end

    for i = 0, npieces do
      init_mesh_zones(
        rz_all_p[i], rz_spans[i], nspans_zones)
    end

    for i = 0, npieces do
      calc_centers_full(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans[i], nspans_zones,
        true)
    end

    for i = 0, npieces do
      calc_volumes_full(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans[i], nspans_zones,
        true)
    end

    for i = 0, npieces do
      init_side_fracs(
        rz_all_p[i], rp_all_private_p[i], rp_all_ghost_p[i],
        rs_all_p[i], rs_spans[i], nspans_zones)
    end

    for i = 0, npieces do
      init_hydro(
        rz_all_p[i], rz_spans[i],
        rinit, einit, rinitsub, einitsub,
        subregion_0, subregion_1, subregion_2, subregion_3,
        nspans_zones)
    end

    for i = 0, npieces do
      init_radial_velocity(
        rp_all_private_p[i], rp_spans_private[i],
        uinitradial, nspans_points)
    end
    for i = 0, npieces do
      init_radial_velocity(
        rp_all_shared_p[i], rp_spans_shared[i],
        uinitradial, nspans_points)
    end
  end

  __demand(__spmd)
  do
    -- Main Simulation Loop
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
                     rp_spans_private[i],
                     dt,
                     nspans_points,
                     enable, print_ts)
      end
      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_half(rp_all_shared_p[i],
                     rp_spans_shared[i],
                     dt,
                     nspans_points,
                     enable, print_ts)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        calc_everything(rz_all_p[i],
                        rp_all_private_p[i],
                        rp_all_ghost_p[i],
                        rs_all_p[i],
                        rz_spans[i],
                        rs_spans[i],
                        alfa, gamma, ssmin, dt,
                        q1, q2,
                        nspans_zones,
                        enable)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_full(rp_all_private_p[i],
                     rp_spans_private[i],
                     dt,
                     nspans_points,
                     enable)
      end
      -- __demand(__parallel)
      for i = 0, npieces do
        adv_pos_full(rp_all_shared_p[i],
                     rp_spans_shared[i],
                     dt,
                     nspans_points,
                     enable)
      end

      -- __demand(__parallel)
      for i = 0, npieces do
        calc_everything_full(rz_all_p[i],
                             rp_all_private_p[i],
                             rp_all_ghost_p[i],
                             rs_all_p[i],
                             rz_spans[i],
                             rs_spans[i],
                             dt,
                             nspans_zones,
                             enable)
      end

      print_ts = requested_print_ts and cycle == cstop - 1 - prune

      dthydro = dtmax
      -- __demand(__parallel)
      for i = 0, npieces do
        dthydro min= calc_dt_hydro(rz_all_p[i],
                                   rz_spans[i],
                                   dt, dtmax, cfl, cflv,
                                   nspans_zones,
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
  local link_flags = {"-L" .. root_dir, "-lpennant"}
  local exe = os.getenv('OBJNAME') or "pennant"
  regentlib.saveobj(toplevel, exe, "executable", cpennant.register_mappers, link_flags)
else
  regentlib.start(toplevel, cpennant.register_mappers)
end
