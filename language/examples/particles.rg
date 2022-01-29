-- Copyright 2022 Stanford University
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

import "regent"
philox = require "philox"

local c = regentlib.c

local prng = philox.philox2x32(10)

fspace fs_grid {
  charge : double,
  potential : double,
  { ex, ey, ez } : double,

  -- auxiliary fields used by field CG solver
  { cg_r, cg_p, cg_Ap } : double,
}

fspace fs_parts {
  { mass, charge } : double,
  { px, py, pz } : double,  -- current particle position
  { vx, vy, vz } : double,  -- current particle velocity
  { ex, ey, ez } : double,  -- electric field at current particle location
}

fspace grid_info {
  cells : int3d,  -- number of cells in each direction
  { xmin, ymin, zmin } : double,  -- grid starting location
  --xmin : double, ymin : double, zmin : double,
  { dx, dy, dz } : double,  -- cell size in each dimension
  --dx : double, dy : double, dz : double,
}

fspace grid_parts(grid : region(ispace(int3d), fs_grid), pspace : ispace(int3d)) {
  { main, xm, xp, ym, yp, zm, zp } : partition(disjoint, grid, pspace)
}

terra add_colored_rect(coloring : c.legion_domain_point_coloring_t,
                       color : int3d, lo : int3d, hi : int3d)
  var rect = c.legion_rect_3d_t { lo = lo:to_point(), hi = hi:to_point() }
  c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(),
                                              c.legion_domain_from_rect_3d(rect))
end

task partition_grid(grid : region(ispace(int3d), fs_grid), pspace : ispace(int3d), depth : int32, wrap : bool)
  var coloring_main = c.legion_domain_point_coloring_create()
  var coloring_xm = c.legion_domain_point_coloring_create()
  var coloring_xp = c.legion_domain_point_coloring_create()
  var coloring_ym = c.legion_domain_point_coloring_create()
  var coloring_yp = c.legion_domain_point_coloring_create()
  var coloring_zm = c.legion_domain_point_coloring_create()
  var coloring_zp = c.legion_domain_point_coloring_create()

  var gbounds = grid.bounds
  var pbounds = pspace.bounds

  var gsize = gbounds:size()
  var psize = pbounds:size()

  for p in pspace do
    var xlo : int32 = gbounds.lo.x + ((p.x - pbounds.lo.x) * gsize.x / psize.x)
    var xhi : int32 = gbounds.lo.x + ((p.x - pbounds.lo.x + 1) * gsize.x / psize.x) - 1
    var ylo : int32 = gbounds.lo.y + ((p.y - pbounds.lo.y) * gsize.y / psize.y)
    var yhi : int32 = gbounds.lo.y + ((p.y - pbounds.lo.y + 1) * gsize.y / psize.y) - 1
    var zlo : int32 = gbounds.lo.z + ((p.z - pbounds.lo.z) * gsize.z / psize.z)
    var zhi : int32 = gbounds.lo.z + ((p.z - pbounds.lo.z + 1) * gsize.z / psize.z) - 1
    add_colored_rect(coloring_main, p, { x = xlo, y = ylo, z = zlo }, { x = xhi, y = yhi, z = zhi })
    --c.printf("p=(%d,%d,%d), lo=(%d,%d,%d), hi=(%d,%d,%d)\n", p.x,p.y,p.z,xlo,ylo,zlo,xhi,yhi,zhi)

    -- x minus
    do
      var xmhi : int32 = xlo - 1
      var xmlo : int32 = xmhi - (depth - 1)
      if wrap then
        -- todo
      else
        xmlo max= gbounds.lo.x
      end
      add_colored_rect(coloring_xm, p, { x = xmlo, y = ylo, z = zlo }, { x = xmhi, y = yhi, z = zhi })
    end

    -- x plus
    do
      var xplo : int32 = xhi + 1
      var xphi : int32 = xplo + (depth - 1)
      if wrap then
        -- todo
      else
        xphi min= gbounds.hi.x
      end
      add_colored_rect(coloring_xp, p, { x = xplo, y = ylo, z = zlo }, { x = xphi, y = yhi, z = zhi })
    end

    -- y minus
    do
      var ymhi : int32 = ylo - 1
      var ymlo : int32 = ymhi - (depth - 1)
      if wrap then
        -- todo
      else
        ymlo max= gbounds.lo.y
      end
      add_colored_rect(coloring_ym, p, { x = xlo, y = ymlo, z = zlo }, { x = xhi, y = ymhi, z = zhi })
    end

    -- y plus
    do
      var yplo : int32 = yhi + 1
      var yphi : int32 = yplo + (depth - 1)
      if wrap then
        -- todo
      else
        yphi min= gbounds.hi.y
      end
      add_colored_rect(coloring_yp, p, { x = xlo, y = yplo, z = zlo }, { x = xhi, y = yphi, z = zhi })
    end

    -- x minus
    do
      var zmhi : int32 = zlo - 1
      var zmlo : int32 = zmhi - (depth - 1)
      if wrap then
        -- todo
      else
        zmlo max= gbounds.lo.z
      end
      add_colored_rect(coloring_zm, p, { x = xlo, y = ylo, z = zmlo }, { x = xhi, y = yhi, z = zmhi })
    end

    -- z plus
    do
      var zplo : int32 = zhi + 1
      var zphi : int32 = zplo + (depth - 1)
      if wrap then
        -- todo
      else
        zphi min= gbounds.hi.z
      end
      add_colored_rect(coloring_zp, p, { x = xlo, y = ylo, z = zplo }, { x = xhi, y = yhi, z = zphi })
    end
  end

  var gp = [grid_parts(grid, pspace)]{
    main = partition(disjoint, grid, coloring_main, pspace),
    xm = partition(disjoint, grid, coloring_xm, pspace),
    xp = partition(disjoint, grid, coloring_xp, pspace),
    ym = partition(disjoint, grid, coloring_ym, pspace),
    yp = partition(disjoint, grid, coloring_yp, pspace),
    zm = partition(disjoint, grid, coloring_zm, pspace),
    zp = partition(disjoint, grid, coloring_zp, pspace),
  }

  c.legion_domain_point_coloring_destroy(coloring_main)
  c.legion_domain_point_coloring_destroy(coloring_xm)
  c.legion_domain_point_coloring_destroy(coloring_xp)
  c.legion_domain_point_coloring_destroy(coloring_ym)
  c.legion_domain_point_coloring_destroy(coloring_yp)
  c.legion_domain_point_coloring_destroy(coloring_zm)
  c.legion_domain_point_coloring_destroy(coloring_zp)

  return gp
end

task print_charge(grid : region(ispace(int3d), fs_grid))
where reads(grid.charge)
do
  var b = grid.bounds
  c.printf("print_charge: (%d,%d,%d) -> (%d,%d,%d)\n", b.lo.x, b.lo.y, b.lo.z, b.hi.x, b.hi.y, b.hi.z)
  for i in grid do
     if (grid[i].charge ~= 0) then c.printf("charge @ (%d,%d,%d) = %g\n", i.x, i.y, i.z, grid[i].charge) end
  end
end

function gen_print(label, is, fs, f)
  local task print(r : region(ispace(int3d), fs))
  where reads(r.[f])
  do
    for i in r do
       if (r[i].[f] ~= 0) then c.printf("%s @ (%d,%d,%d) = %g\n", label, i.x, i.y, i.z, r[i].[f]) end
    end
  end
  print:set_name("print_" .. label)
  return print
end

task init_random_particles(particles : region(ispace(int1d), fs_parts),
                           seed : uint32, ginfo : grid_info)
where writes(particles.{mass,charge,px,py,pz,vx,vy,vz})
do
  var xsize : double = (ginfo.cells.x - 1) * ginfo.dx
  var ysize : double = (ginfo.cells.y - 1) * ginfo.dy
  var zsize : double = (ginfo.cells.z - 1) * ginfo.dz

  for i in particles do 
    particles[i].mass   = 1 + philox.u64tofp(prng(seed, [uint64](i) * 10 + 0))
    particles[i].charge = 10 -- -1 + 2*philox.u64tofp(prng(seed, [uint64](i) * 10 + 1))
    particles[i].px     = ginfo.xmin + xsize * philox.u64tofp(prng(seed, [uint64](i) * 10 + 2))
    particles[i].py     = ginfo.ymin + ysize * philox.u64tofp(prng(seed, [uint64](i) * 10 + 3))
    particles[i].pz     = ginfo.zmin + zsize * philox.u64tofp(prng(seed, [uint64](i) * 10 + 4))
    particles[i].vx = ginfo.dx * (-0.5 + philox.u64tofp(prng(seed, [uint64](i) * 10 + 5)))
    particles[i].vy = ginfo.dy * (-0.5 + philox.u64tofp(prng(seed, [uint64](i) * 10 + 6)))
    particles[i].vz = ginfo.dz * (-0.5 + philox.u64tofp(prng(seed, [uint64](i) * 10 + 7)))
  end
end

task print_particles(particles : region(ispace(int1d), fs_parts))
where reads writes(particles.{mass,charge,px,py,pz,vx,vy,vz})
do
  --var b = particles.ispace.impl
  --c.printf("%d %d\n", b.dim,4)--lo.x[0], b.hi.x[0])
  for i in particles do 
    c.printf("particle #%d: m=%g c=%g p=(%g,%g,%g) v=(%g,%g,%g)\n",
	     i, particles[i].mass, particles[i].charge,
	     particles[i].px, particles[i].py, particles[i].pz,
	     particles[i].vx, particles[i].vy, particles[i].vz)
  end
end

task splat_charge(grid : region(ispace(int3d), fs_grid), 
		  particles : region(ispace(int1d), fs_parts),
                  ginfo : grid_info)
where reads(particles.{charge,px,py,pz}), reduces +(grid.charge)
do
  -- get the bounds of our part of the grid - we'll only contribute charge to pieces we own
  var gbounds = grid.bounds

  for p in particles do
    var cx = min(max((particles[p].px - ginfo.xmin) / ginfo.dx, 0), ginfo.cells.x - 1)
    var xlo = [int](cx)
    var xhi = min(xlo + 1, ginfo.cells.x - 1)
    var xfrac = cx - xlo

    var cy = min(max((particles[p].py - ginfo.ymin) / ginfo.dy, 0), ginfo.cells.y - 1)
    var ylo = [int](cy)
    var yhi = min(ylo + 1, ginfo.cells.y - 1)
    var yfrac = cy - ylo

    var cz = min(max((particles[p].pz - ginfo.zmin) / ginfo.dz, 0), ginfo.cells.z - 1)
    var zlo = [int](cz)
    var zhi = min(zlo + 1, ginfo.cells.z - 1)
    var zfrac = cz - zlo

    if (xlo >= gbounds.lo.x) and (xlo <= gbounds.hi.x) then
      if (ylo >= gbounds.lo.y) and (ylo <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
          grid[ { x = xlo, y = ylo, z = zlo } ].charge += particles[p].charge * (1 - xfrac) * (1 - yfrac) * (1 - zfrac)
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
          grid[ { x = xlo, y = ylo, z = zhi } ].charge += particles[p].charge * (1 - xfrac) * (1 - yfrac) * (    zfrac)
        end
      end

      if (yhi >= gbounds.lo.y) and (yhi <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
          grid[ { x = xlo, y = yhi, z = zlo } ].charge += particles[p].charge * (1 - xfrac) * (    yfrac) * (1 - zfrac)
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
          grid[ { x = xlo, y = yhi, z = zhi } ].charge += particles[p].charge * (1 - xfrac) * (    yfrac) * (    zfrac)
        end
      end
    end

    if (xhi >= gbounds.lo.x) and (xhi <= gbounds.hi.x) then
      if (ylo >= gbounds.lo.y) and (ylo <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
          grid[ { x = xhi, y = ylo, z = zlo } ].charge += particles[p].charge * (    xfrac) * (1 - yfrac) * (1 - zfrac)
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
          grid[ { x = xhi, y = ylo, z = zhi } ].charge += particles[p].charge * (    xfrac) * (1 - yfrac) * (    zfrac)
        end
      end

      if (yhi >= gbounds.lo.y) and (yhi <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
          grid[ { x = xhi, y = yhi, z = zlo } ].charge += particles[p].charge * (    xfrac) * (    yfrac) * (1 - zfrac)
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
          grid[ { x = xhi, y = yhi, z = zhi } ].charge += particles[p].charge * (    xfrac) * (    yfrac) * (    zfrac)
        end
      end
    end
  end
end

task interpolate_field(grid : region(ispace(int3d), fs_grid), 
		       particles : region(ispace(int1d), fs_parts),
                       ginfo : grid_info)
where reads(grid.{ex,ey,ez}), reads(particles.{px,py,pz}), reduces +(particles.{ex,ey,ez})
do
  -- get the bounds of our part of the grid - we'll only contribute charge to pieces we own
  var gbounds = grid.bounds

  for p in particles do
    var cx = min(max((particles[p].px - ginfo.xmin) / ginfo.dx, 0), ginfo.cells.x - 1)
    var xlo = [int](cx)
    var xhi = min(xlo + 1, ginfo.cells.x - 1)
    var xfrac = cx - xlo

    var cy = min(max((particles[p].py - ginfo.ymin) / ginfo.dy, 0), ginfo.cells.y - 1)
    var ylo = [int](cy)
    var yhi = min(ylo + 1, ginfo.cells.y - 1)
    var yfrac = cy - ylo

    var cz = min(max((particles[p].pz - ginfo.zmin) / ginfo.dz, 0), ginfo.cells.z - 1)
    var zlo = [int](cz)
    var zhi = min(zlo + 1, ginfo.cells.z - 1)
    var zfrac = cz - zlo

    if (xlo >= gbounds.lo.x) and (xlo <= gbounds.hi.x) then
      if (ylo >= gbounds.lo.y) and (ylo <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
	  var pt : int3d = { x = xlo, y = ylo, z = xlo }
          var wt : double = (1 - xfrac) * (1 - yfrac) * (1 - zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
	  var pt : int3d = { x = xlo, y = ylo, z = xhi }
          var wt : double = (1 - xfrac) * (1 - yfrac) * (    zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
      end

      if (yhi >= gbounds.lo.y) and (yhi <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
	  var pt : int3d = { x = xlo, y = yhi, z = xlo }
          var wt : double = (1 - xfrac) * (    yfrac) * (1 - zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
	  var pt : int3d = { x = xlo, y = yhi, z = xhi }
          var wt : double = (1 - xfrac) * (    yfrac) * (    zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
      end
    end

    if (xhi >= gbounds.lo.x) and (xhi <= gbounds.hi.x) then
      if (ylo >= gbounds.lo.y) and (ylo <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
	  var pt : int3d = { x = xhi, y = ylo, z = xlo }
          var wt : double = (    xfrac) * (1 - yfrac) * (1 - zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
	  var pt : int3d = { x = xhi, y = ylo, z = xhi }
          var wt : double = (    xfrac) * (1 - yfrac) * (    zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
      end

      if (yhi >= gbounds.lo.y) and (yhi <= gbounds.hi.y) then
        if (zlo >= gbounds.lo.z) and (zlo <= gbounds.hi.z) then
	  var pt : int3d = { x = xhi, y = yhi, z = xlo }
          var wt : double = (    xfrac) * (    yfrac) * (1 - zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
        if (zhi >= gbounds.lo.z) and (zhi <= gbounds.hi.z) then
	  var pt : int3d = { x = xhi, y = yhi, z = xhi }
          var wt : double = (    xfrac) * (    yfrac) * (    zfrac)
	  particles[p].ex += wt * grid[pt].ex
	  particles[p].ey += wt * grid[pt].ey
	  particles[p].ez += wt * grid[pt].ez
        end
      end
    end
  end
end

task advance_particles(particles : region(ispace(int1d), fs_parts), dt : double)
where reads(particles.{mass,charge,ex,ey,ez}), reads writes(particles.{px,py,pz,vx,vy,vz})
do
  for p in particles do
    var qom : double = particles[p].charge / particles[p].mass
    var ax : double = qom * particles[p].ex
    var ay : double = qom * particles[p].ey
    var az : double = qom * particles[p].ez
    particles[p].px += dt * (particles[p].vx + 0.5 * dt * ax)
    particles[p].py += dt * (particles[p].vy + 0.5 * dt * ay)
    particles[p].pz += dt * (particles[p].vz + 0.5 * dt * az)
    particles[p].vx += dt * ax
    particles[p].vy += dt * ay
    particles[p].vz += dt * az
  end
end

function gen_dotproduct(is, fs, f1, f2)
  local task dotproduct(r : region(is, fs)) : double
  where reads(r.[ terralib.newlist({ f1, f2 }) ])
  do
    var total : double = 0
    for i in r do
      total += r[i].[f1] * r[i].[f2]
    end
    return total
  end
  dotproduct:set_name("dotproduct_" .. f1 .. "_" .. f2)
  return dotproduct
end

task poisson_spmv(grid : region(ispace(int3d), fs_grid), ginfo : grid_info)
where reads(grid.cg_p), writes(grid.cg_Ap)
do
  for i in grid do
    var dxm2 : double = 1.0 / (ginfo.dx * ginfo.dx)
    var dym2 : double = 1.0 / (ginfo.dy * ginfo.dy)
    var dzm2 : double = 1.0 / (ginfo.dz * ginfo.dz)
    var k = 2 * (dxm2 + dym2 + dzm2) * grid[i].cg_p
    
    if(i.x == 0) then
       --k -= grid[ { x = ginfo.cells.x - 1, y = i.y, z = i.z } ].cg_p
    else
       k -= dxm2 * grid[ { x = i.x - 1, y = i.y, z = i.z } ].cg_p
    end
    if(i.x == ginfo.cells.x - 1) then
       --k -= grid[ { x = 0, y = i.y, z = i.z } ].cg_p
    else
       k -= dxm2 * grid[ { x = i.x + 1, y = i.y, z = i.z } ].cg_p
    end
    if(i.y == 0) then
       --k -= grid[ { x = i.x, y = ginfo.cells.y - 1, z = i.z } ].cg_p
    else
       k -= dym2 * grid[ { x = i.x, y = i.y - 1, z = i.z } ].cg_p
    end
    if(i.y == ginfo.cells.y - 1) then
       --k -= grid[ { x = i.x, y = 0, z = i.z } ].cg_p
    else
       k -= dym2 * grid[ { x = i.x, y = i.y + 1, z = i.z } ].cg_p
    end
    if(i.z == 0) then
       --k -= grid[ { x = i.x, y = i.y, z = ginfo.cells.z - 1 } ].cg_p
    else
       k -= dzm2 * grid[ { x = i.x, y = i.y, z = i.z - 1 } ].cg_p
    end
    if(i.z == ginfo.cells.z - 1) then
       --k -= grid[ { x = i.x, y = i.y, z = 0 } ].cg_p
    else
       k -= dzm2 * grid[ { x = i.x, y = i.y, z = i.z + 1 } ].cg_p
    end
    grid[i].cg_Ap = k
  end
end

task poisson_update_x(grid : region(ispace(int3d), fs_grid), alpha : double)
where reads(grid.cg_p), reads writes(grid.potential)
do
  for i in grid do
    grid[i].potential += alpha * grid[i].cg_p
  end
end

task poisson_update_r(grid : region(ispace(int3d), fs_grid), alpha : double)
where reads(grid.cg_Ap), reads writes(grid.cg_r)
do
  for i in grid do
    grid[i].cg_r -= alpha * grid[i].cg_Ap
  end
end

task poisson_update_p(grid : region(ispace(int3d), fs_grid), beta : double)
where reads(grid.cg_r), reads writes(grid.cg_p)
do
  for i in grid do
    grid[i].cg_p = grid[i].cg_r + beta * grid[i].cg_p
  end
end

__demand(__inline)
task poisson_solve(grid : region(ispace(int3d), fs_grid),
                   pspace : ispace(int3d),
                   gparts : grid_parts(grid, pspace),
                   ginfo : grid_info)
where reads(grid.charge), reads writes(grid.potential), reads writes(grid.{cg_r,cg_p,cg_Ap})
do
  -- this is a vanilla conjugate gradient solver using a stencil as the implicit matrix A
  fill(grid.potential, 0)
  copy(grid.charge, grid.cg_r)
  copy(grid.cg_r, grid.cg_p)

  var res : double = [ gen_dotproduct(ispace(int3d), fs_grid, "cg_r", "cg_r") ](grid)
  var maxiters = 2*(ginfo.cells.x + ginfo.cells.y + ginfo.cells.z)
  var iter = 1
  while iter <= maxiters do
    poisson_spmv(grid, ginfo)
    var ptAp : double = [ gen_dotproduct(ispace(int3d), fs_grid, "cg_p", "cg_Ap") ](grid)
    var alpha : double = res / ptAp
    poisson_update_x(grid, alpha)
    poisson_update_r(grid, alpha)
    var newres : double = [ gen_dotproduct(ispace(int3d), fs_grid, "cg_r", "cg_r") ](grid)
    if (newres < 1e-10) then break end
    var beta : double = newres / res
    poisson_update_p(grid, beta)
    res = newres
    iter += 1    
  end
  if iter <= maxiters then
    c.printf("converged after %d iterations\n", iter)
  else
    c.printf("failed to converge after %d iterations!\n", maxiters)
  end
end

task calculate_field(grid : region(ispace(int3d), fs_grid), ginfo : grid_info)
where reads(grid.potential), writes(grid.{ex,ey,ez})
do
  var gbounds = grid.bounds

  for i in grid do
    do
      var dp : double = 0
      if (i.x > gbounds.lo.x) then dp -= grid[ { x = i.x - 1, y = i.y, z = i.z } ].potential end
      if (i.x < gbounds.hi.x) then dp += grid[ { x = i.x + 1, y = i.y, z = i.z } ].potential end
      grid[i].ex = dp / ginfo.dx
    end
    do
      var dp : double = 0
      if (i.y > gbounds.lo.y) then dp -= grid[ { x = i.x, y = i.y - 1, z = i.z } ].potential end
      if (i.y < gbounds.hi.y) then dp += grid[ { x = i.x, y = i.y + 1, z = i.z } ].potential end
      grid[i].ey = dp / ginfo.dy
    end
    do
      var dp : double = 0
      if (i.z > gbounds.lo.z) then dp -= grid[ { x = i.x, y = i.y, z = i.z - 1 } ].potential end
      if (i.z < gbounds.hi.z) then dp += grid[ { x = i.x, y = i.y, z = i.z + 1 } ].potential end
      grid[i].ez = dp / ginfo.dz
    end
  end
end

task main()
  var ginfo : grid_info = [grid_info]{ cells = { x = 8, y = 8, z = 8 },
			    xmin = 0, ymin = 0, zmin = 0,
                            dx = 1e-5, dy = 1e-5, dz = 1e-5,
			  }
  var nblks : int3d = { x = 2, y = 2, z = 2 }
  var nparts : int64 = 10
  var npblks : int64 = 2
  var seed : uint32 = 12345
  var dt : double = 1
  var steps : int = 10

  var is_grid = ispace(int3d, ginfo.cells)
  var grid = region(is_grid, fs_grid)

  var ps_grid = ispace(int3d, nblks)
  var gparts = partition_grid(grid, ps_grid, 1, false)

  var is_parts = ispace(int1d, nparts)
  var particles = region(is_parts, fs_parts)

  init_random_particles(particles, seed, ginfo)

  for i = 0,steps do
    fill(grid.charge, 0)
    splat_charge(grid, particles, ginfo)

    --print_charge(grid)

    poisson_solve(grid, ps_grid, gparts, ginfo)

    calculate_field(grid, ginfo)

    --;[ gen_print("potential", is_grid, fs_grid, "potential") ](grid)
    --;[ gen_print("charge", is_grid, fs_grid, "charge") ](grid)
  
    --print_particles(particles)

    fill(particles.{ex,ey,ez}, 0)
    interpolate_field(grid, particles, ginfo)

    c.printf("step %d\n", i)
    advance_particles(particles, dt)
  end

  print_particles(particles)
end

regentlib.start(main)
