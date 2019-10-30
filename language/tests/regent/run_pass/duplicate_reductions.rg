-- This test comes from issue #215
-- It tests for duplicate applications of reductions
-- in the case where we do a bunch of reductions followed
-- by a bunch of reads on an aliased partition. It's 
-- important that the runtime deduplicate the reductions to the 
-- same instance for overlapping but non-interfering readers

-- mesh: 17x17 vertex-centered
-- multigrid 2 level
-- convergence: 4 orders residual reduction
-- GS_RB: 20 interations 2.342333e-01
-- Jacobi: 75 2.319495e-01
-- SOR_RB: 8 1.410193e-01

import "regent"

local c = regentlib.c
local cmath = terralib.includec("math.h")
local sqrt  = regentlib.sqrt(double)
local coloring = {}
coloring.create = c.legion_domain_point_coloring_create
coloring.destroy = c.legion_domain_point_coloring_destroy
coloring.color_domain = c.legion_domain_point_coloring_color_domain

fspace Node
{
  -- x   : double; -- x coordinate (need?)
  -- y   : double; -- y coordinate (need?)
  phi : double; -- solution
  -- phi_exact : double; -- exact solution
  f   : double; -- source term
  aux : double; -- residual, phi_update, ...
  red : bool;   -- black-red ordering
}

task initialize(r_mesh  : region(ispace(int2d), Node),
                n_nodes : uint32, 
                level   : uint32, 
                n_part  : uint32, 
                color   : int2d)
where
  reads writes(r_mesh)
do
  -- assign coordinates
  var x_init : uint32 = color.x * (n_nodes-1)/n_part + 1
  var y_init : uint32 = color.y * (n_nodes-1)/n_part + 1
  if color.x == 0 then
    x_init -= 1
  end
  if color.y == 0 then
    y_init -= 1
  end
  var limits = r_mesh.bounds
  var ii : uint32 = x_init
  var jj : uint32 = y_init
  for i = [int](limits.lo.x), [int](limits.hi.x) + 1 do
    for j = [int](limits.lo.y), [int](limits.hi.y) + 1 do
      -- assign coordinates
      -- r_mesh[{i, j}].x = [double](ii) / [double](n_nodes-1)
      -- r_mesh[{i, j}].y = [double](jj) / [double](n_nodes-1)
      var x : double = [double](ii) / [double](n_nodes-1)
      var y : double = [double](jj) / [double](n_nodes-1)
      if level == 0 then
        -- assign initial value of phi
        if ii == 0 or jj == 0 or ii == (n_nodes-1) or jj == (n_nodes-1) then
          r_mesh[{i, j}].phi = cmath.exp(x) * cmath.exp(-2.0*y)
        else
          r_mesh[{i, j}].phi = 0
        end
        -- assign phi_exact
        -- r_mesh[{i, j}].phi_exact = cmath.exp(r_mesh[{i, j}].x) * cmath.exp(-2.0*r_mesh[{i, j}].y)
        -- assign source term f
        r_mesh[{i, j}].f = -5.0 * cmath.exp(x) * cmath.exp(-2.0*y)
      else
        r_mesh[{i,j}].phi = 0
        r_mesh[{i,j}].f = 0
      end
      -- assign residual
      r_mesh[{i,j}].aux = 0
      -- assign color
      r_mesh[{i,j}].red = false
      if (ii%2==0 and jj%2==0) or (ii%2==1 and jj%2==1) then
        r_mesh[{i,j}].red = true
      end
      --c.printf("{%d, %d}={%f, %f} phi= %f f=%f red=%d\n", 
      --  ii, jj, x, y, r_mesh[{i, j}].phi, r_mesh[{i, j}].f, r_mesh[{i, j}].red)
      jj += 1
    end
    ii += 1
    jj = y_init
  end

end

task create_interior_partition(r_mesh : region(ispace(int2d), Node))
  var coloring = c.legion_domain_coloring_create()
  var bounds = r_mesh.ispace.bounds
  c.legion_domain_coloring_color_domain(coloring, 0,
    rect2d { bounds.lo + {1, 1}, bounds.hi - {1, 1} })
  var interior_mesh_partition = partition(disjoint, r_mesh, coloring)
  c.legion_domain_coloring_destroy(coloring)
  return interior_mesh_partition
end

task compute_residual(r_mesh     : region(ispace(int2d), Node),
                      r_interior : region(ispace(int2d), Node),
                      n_nodes    : uint32)
where
  reads(r_mesh.phi, r_interior.f),
  writes(r_interior.aux)
do
  var h2 : double = 1.0 / ([double](n_nodes-1)*(n_nodes-1))
  var norm : double = 0
  for e in r_interior do
    var result : double = r_interior[e].f + (r_mesh[e+{0,1}].phi+r_mesh[e+{1,0}].phi+
      r_mesh[e+{0,-1}].phi+r_mesh[e+{-1,0}].phi-4*r_mesh[e].phi) / h2
    r_interior[e].aux = result
    norm += result * result
  end
  return norm
end

task jacobi(r_mesh     : region(ispace(int2d), Node),
            r_interior : region(ispace(int2d), Node),
            n_nodes    : uint32)
where
  reads(r_mesh.phi, r_interior.f),
  writes(r_interior.aux)
do
  var h2 : double = 1.0 / ([double](n_nodes-1)*(n_nodes-1))
  for e in r_interior do
    r_interior[e].aux = (r_mesh[e+{1,0}].phi+r_mesh[e+{-1,0}].phi+
      r_mesh[e+{0,1}].phi+r_mesh[e+{0,-1}].phi+h2*r_interior[e].f) / 4
  end
end

task update_phi(r_interior : region(ispace(int2d), Node),
                red        : bool,
                black      : bool)
where
  reads(r_interior.aux, r_interior.red),
  writes(r_interior.phi)
do
  if red then
    for e in r_interior do
      if e.red then e.phi = e.aux end
    end
  elseif black then
    for e in r_interior do
      if not e.red then e.phi = e.aux end
    end
  elseif not (red or black) then
    for e in r_interior do
      e.phi = e.aux
    end   
  else 
    c.printf("Error\n")
  end
end

-- task gauss_seidel(r_mesh     : region(ispace(int2d), Node),
--                   r_interior : region(ispace(int2d), Node),
--                   n_nodes    : uint32)
-- where
--   reads(r_mesh.phi, r_interior.f),
--   writes(r_mesh.phi)
-- do
--   var h2 : double = 1.0 / ([double](n_nodes-1)*(n_nodes-1))
--   var limits = r_interior.bounds
--   for i = [int](limits.lo.x), [int](limits.hi.x) + 1 do
--     for j = [int](limits.lo.y), [int](limits.hi.y) + 1 do
--     r_mesh[{i,j}].phi = (r_mesh[{i+1,j}].phi+r_mesh[{i-1,j}].phi+
--       r_mesh[{i,j+1}].phi+r_mesh[{i,j-1}].phi+h2*r_interior[{i,j}].f) / 4
--     end
--   end
-- end

task gauss_seidel_RB(r_mesh     : region(ispace(int2d), Node),
                     r_interior : region(ispace(int2d), Node),
                     n_nodes    : uint32,
                     red        : bool)
where
  reads(r_mesh.phi, r_interior.f, r_interior.red),
  writes(r_interior.aux)
do
  var h2 : double = 1.0 / ([double](n_nodes-1)*(n_nodes-1))
  for e in r_interior do
    if (red and e.red) or (not (red or e.red)) then
      r_interior[e].aux = (r_mesh[e+{1,0}].phi+r_mesh[e+{-1,0}].phi+
        r_mesh[e+{0,1}].phi+r_mesh[e+{0,-1}].phi+h2*r_interior[e].f) / 4
    end
  end
end

task SOR_RB(r_mesh     : region(ispace(int2d), Node),
            r_interior : region(ispace(int2d), Node),
            n_nodes    : uint32,
            red        : bool,
            relax      : double)
where
  reads(r_mesh.phi, r_interior.phi, r_interior.f, r_interior.red),
  writes(r_interior.aux)
do
  var h2 : double = 1.0 / ([double](n_nodes-1)*(n_nodes-1))
  for e in r_interior do
    if (red and e.red) or (not (red or e.red)) then
      r_interior[e].aux = (1-relax) * r_interior[e].phi + relax * (r_mesh[e+{1,0}].phi+
        r_mesh[e+{-1,0}].phi+ r_mesh[e+{0,1}].phi+r_mesh[e+{0,-1}].phi+h2*r_interior[e].f) / 4
    end
  end
end

--__demand(__inline)
task smooth_top(r_mesh     : region(ispace(int2d), Node),
                p_private  : partition(disjoint, r_mesh, ispace(int2d)),
                p_halo     : partition(aliased, r_mesh, ispace(int2d)),
                n_nodes    : uint32,
                smoother   : uint32,
                relax      : double, 
                sweeps     : uint32)
where
  --reads writes(r_mesh.phi, r_mesh.aux),
  --reads(r_mesh.f, r_mesh.red)
  reads writes(r_mesh)
do
  for i = 0, sweeps do
    if smoother == 0 then
      for color in p_private.colors do
        jacobi(p_halo[color], p_private[color], n_nodes)
      end
      for color in p_private.colors do
        update_phi(p_private[color], false, false)
      end
    elseif smoother == 1 then
      for color in p_private.colors do
        gauss_seidel_RB(p_halo[color], p_private[color], n_nodes, true)
      end
      for color in p_private.colors do
        update_phi(p_private[color], true, false)
      end
      for color in p_private.colors do
        gauss_seidel_RB(p_halo[color], p_private[color], n_nodes, false)
      end
      for color in p_private.colors do
        update_phi(p_private[color], false, true)
      end
    elseif smoother == 2 then
      for color in p_private.colors do
        SOR_RB(p_halo[color], p_private[color], n_nodes, true, relax)
      end
      for color in p_private.colors do
        update_phi(p_private[color], true, false)
      end
      for color in p_private.colors do
        SOR_RB(p_halo[color], p_private[color], n_nodes, false, relax)
      end
      for color in p_private.colors do
        update_phi(p_private[color], false, true)
      end
    end
  end
end

task perform_restriction(r_mesh_0     : region(ispace(int2d), Node),
                         r_interior_1 : region(ispace(int2d), Node))
where
  reads(r_mesh_0.aux),
  writes(r_interior_1.f, r_interior_1.phi)
do
  var base0 = r_mesh_0.ispace.bounds.lo + {1,1}
  var lim1 = r_interior_1.ispace.bounds
  var ii : uint32 = 0
  var jj : uint32 = 0
  for i = [int](lim1.lo.x), [int](lim1.hi.x) + 1 do
    for j = [int](lim1.lo.y), [int](lim1.hi.y) + 1 do
      r_interior_1[{i,j}].phi = 0
      r_interior_1[{i,j}].f =   r_mesh_0[base0+{2*ii,2*jj}].aux/16.0
                              + r_mesh_0[base0+{2*ii,2*jj+2}].aux/16.0
                              + r_mesh_0[base0+{2*ii+2,2*jj}].aux/16.0
                              + r_mesh_0[base0+{2*ii+2,2*jj+2}].aux/16.0
                              + r_mesh_0[base0+{2*ii+1,2*jj}].aux/8.0
                              + r_mesh_0[base0+{2*ii+1,2*jj+2}].aux/8.0
                              + r_mesh_0[base0+{2*ii,2*jj+1}].aux/8.0
                              + r_mesh_0[base0+{2*ii+2,2*jj+1}].aux/8.0
                              + r_mesh_0[base0+{2*ii+1,2*jj+1}].aux/4.0
      jj += 1
    end
    jj = 0
    ii += 1
  end
end

task prolongate(r_mesh_0     : region(ispace(int2d), Node),
                r_interior_1 : region(ispace(int2d), Node))
where
  reads(r_interior_1.phi),
  reduces+(r_mesh_0.phi)
do
  var base0 = r_mesh_0.ispace.bounds.lo + {1,1}
  var lim1 = r_interior_1.ispace.bounds
  var ii : uint32 = 0
  var jj : uint32 = 0
  for i = [int](lim1.lo.x), [int](lim1.hi.x) + 1 do
    for j = [int](lim1.lo.y), [int](lim1.hi.y) + 1 do
      r_mesh_0[base0+{2*ii,2*jj}].phi += r_interior_1[{i,j}].phi/4.0
      r_mesh_0[base0+{2*ii+2,2*jj}].phi += r_interior_1[{i,j}].phi/4.0
      r_mesh_0[base0+{2*ii,2*jj+2}].phi += r_interior_1[{i,j}].phi/4.0
      r_mesh_0[base0+{2*ii+2,2*jj+2}].phi += r_interior_1[{i,j}].phi/4.0
      r_mesh_0[base0+{2*ii+1,2*jj}].phi += r_interior_1[{i,j}].phi/2.0
      r_mesh_0[base0+{2*ii+1,2*jj+2}].phi += r_interior_1[{i,j}].phi/2.0
      r_mesh_0[base0+{2*ii,2*jj+1}].phi += r_interior_1[{i,j}].phi/2.0
      r_mesh_0[base0+{2*ii+2,2*jj+1}].phi += r_interior_1[{i,j}].phi/2.0
      r_mesh_0[base0+{2*ii+1,2*jj+1}].phi += r_interior_1[{i,j}].phi
      jj += 1
    end
    jj = 0
    ii += 1
  end
end

task block_task(r_mesh : region(ispace(int2d), Node))
where
  reads writes(r_mesh)
do
  return 1
end

terra wait_for(x : int) return 1 end

task main()
  -- the number of nodes in x and y directions
  var n_nodes : uint32 = 17
  -- the number of multigrid levels
  var n_levels : uint32 = 2
  -- the number of partitions in each dimension
  var n_part  : unit32 = 2
  -- convergence criteria
  var tolerance : double = 4.0
  -- Choice of iterative smoother
  -- 0: Jacobi
  -- 1: Gauss-Seidel
  -- 2: SOR
  var smoother : uint32 = 2
  -- number of smoothing sweeps
  var sweeps : uint32 = 2
  -- relaxation parameter for SOR
  var relax : double = 1.5
  -- 2D mesh region (TODO: metaprogramming)
  var r_mesh_0 = region(ispace(int2d, {n_nodes, n_nodes}), Node)
  var r_mesh_1 = region(ispace(int2d, {(n_nodes-1)/2+1, (n_nodes-1)/2+1}), Node)
  -- Create a sub-region for the interior part of mesh (TODO: metaprogramming)
  var p_interior_0 = create_interior_partition(r_mesh_0)
  var r_interior_0 = p_interior_0[0]
  var p_interior_1 = create_interior_partition(r_mesh_1)
  var r_interior_1 = p_interior_1[0]

  -- Partition r_interior (TODO: metaprogramming)
  -- 2^m+1 x 2^m+1 grid, 4^k total subregions (TODO: relax this)
  var p_private_colors = ispace(int2d, {n_part, n_part})
  var n_nodes_tmp : uint32 = n_nodes
  var interior_base = r_interior_0.bounds.lo
  var sub_size : uint32 = (n_nodes_tmp-1)/n_part
  var c_private_0 = coloring.create()
  for i = 0, (n_nodes_tmp-2), sub_size do
    for j = 0, (n_nodes_tmp-2), sub_size do
      var size_i : uint32 = sub_size
      var size_j : uint32 = sub_size
      if (i/sub_size == (n_part-1)) then
        size_i -= 1
      end
      if (j/sub_size == (n_part-1)) then
        size_j -= 1
      end
      var private_bounds : rect2d = {interior_base + {i, j}, interior_base + {i+size_i-1, j+size_j-1}}
      var color : int2d = [int2d]({i/sub_size, j/sub_size})
      coloring.color_domain(c_private_0, color, private_bounds)
    end
  end
  var p_private_0 = partition(disjoint, r_mesh_0, c_private_0, p_private_colors)
  coloring.destroy(c_private_0)

  n_nodes_tmp = (n_nodes_tmp-1)/2 + 1
  interior_base = r_interior_1.bounds.lo
  sub_size = (n_nodes_tmp-1)/n_part
  var c_private_1 = coloring.create()
  for i = 0, (n_nodes_tmp-2), sub_size do
    for j = 0, (n_nodes_tmp-2), sub_size do
      var size_i : uint32 = sub_size
      var size_j : uint32 = sub_size
      if (i/sub_size == (n_part-1)) then
        size_i -= 1
      end
      if (j/sub_size == (n_part-1)) then
        size_j -= 1
      end
      var private_bounds : rect2d = {interior_base + {i, j}, interior_base + {i+size_i-1, j+size_j-1}}
      var color : int2d = [int2d]({i/sub_size, j/sub_size})
      coloring.color_domain(c_private_1, color, private_bounds)
    end
  end
  var p_private_1 = partition(disjoint, r_mesh_1, c_private_1, p_private_colors)
  coloring.destroy(c_private_1)

  -- Create halo partitions (TODO: metaprogramming)
  var c_halo_0 = coloring.create()
  for color in p_private_colors do
    var bounds = p_private_0[color].bounds
    var halo_bounds : rect2d = {bounds.lo - {1, 1}, bounds.hi + {1, 1}}
    coloring.color_domain(c_halo_0, color, halo_bounds)
  end
  var p_halo_0 = partition(aliased, r_mesh_0, c_halo_0, p_private_colors)
  coloring.destroy(c_halo_0)

  var c_halo_1 = coloring.create()
  for color in p_private_colors do
    var bounds = p_private_1[color].bounds
    var halo_bounds : rect2d = {bounds.lo - {1, 1}, bounds.hi + {1, 1}}
    coloring.color_domain(c_halo_1, color, halo_bounds)
  end
  var p_halo_1 = partition(aliased, r_mesh_1, c_halo_1, p_private_colors)
  coloring.destroy(c_halo_1)

  -- -- Another partition for initialization (TODO: metaprogramming)
  var c_mesh_0 = coloring.create()
  for color in p_private_colors do
    var bounds = p_private_0[color].bounds
    if color.x == 0 then
      bounds.lo.x -= 1
    end
    if color.x == (n_part-1) then
      bounds.hi.x += 1
    end
    if color.y == 0 then
      bounds.lo.y -= 1
    end
    if color.y == (n_part-1) then
      bounds.hi.y += 1
    end
    coloring.color_domain(c_mesh_0, color, bounds)
  end
  var p_mesh_0 = partition(disjoint, r_mesh_0, c_mesh_0, p_private_colors)
  coloring.destroy(c_mesh_0)

  var c_mesh_1 = coloring.create()
  for color in p_private_colors do
    var bounds = p_private_1[color].bounds
    if color.x == 0 then
      bounds.lo.x -= 1
    end
    if color.x == (n_part-1) then
      bounds.hi.x += 1
    end
    if color.y == 0 then
      bounds.lo.y -= 1
    end
    if color.y == (n_part-1) then
      bounds.hi.y += 1
    end
    coloring.color_domain(c_mesh_1, color, bounds)
  end
  var p_mesh_1 = partition(disjoint, r_mesh_1, c_mesh_1, p_private_colors)
  coloring.destroy(c_mesh_1)

  -- initialize the mesh (TODO: metaprogramming)
  for color in p_private_0.colors do
    initialize(p_mesh_0[color], n_nodes, 0, n_part, color)
  end

  for color in p_private_1.colors do
    initialize(p_mesh_1[color], (n_nodes-1)/2+1, 1, n_part, color)
  end

  -- for e in r_mesh_0 do
  --   c.printf("phi=%f f=%f red=%d\n", e.phi, e.f, e.red)
  -- end
  -- for e in r_mesh_1 do
  --   c.printf("phi=%f f=%f red=%d\n", e.phi, e.f, e.red)
  -- end

  -- initialize(r_mesh_0, n_nodes, 0, 0, 0)
  -- initialize(r_mesh_1, (n_nodes-1)/2+1, 1, 0, 0)

  --var token = 0
  --for color in p_private_0.colors do
  --  token += block_task(p_private_0[color])
  --end
  --for color in p_private_1.colors do
  --  token += block_task(p_private_1[color])
  --end
  --wait_for(token)
  --var ts_start = c.legion_get_current_time_in_micros()

  -- compute initial residual
  var residual0 : double = 0
  for color in p_private_0.colors do
    residual0 += compute_residual(p_halo_0[color], p_private_0[color], n_nodes)
  end
  residual0 = sqrt(residual0)
  c.printf("iteration 0, residual %f\n", residual0)
  -- iteration until converge
  var num_iterations = 0
  var converged = false
  while not converged do
    num_iterations += 1
    -- pre-smooth [0]
    smooth_top(r_mesh_0, p_private_0, p_halo_0, n_nodes, smoother, relax, sweeps)
    -- compute residual [0]
    for color in p_private_0.colors do
      compute_residual(p_halo_0[color], p_private_0[color], n_nodes)
    end
    -- restrict residuals to coarser level [0]->[1]
    for color in p_private_1.colors do
      perform_restriction(p_halo_0[color], p_private_1[color])
    end
    
    -- pre-smooth [1]
    smooth_top(r_mesh_1, p_private_1, p_halo_1, (n_nodes-1)/2+1, smoother, relax, sweeps)
    -- post-smooth [1]
    smooth_top(r_mesh_1, p_private_1, p_halo_1, (n_nodes-1)/2+1, smoother, relax, sweeps)
    -- prolongation of correction to fine level [1]->[0]
    for color in p_private_1.colors do
      prolongate(p_halo_0[color], p_private_1[color])
    end

    -- post-smooth [0]
    smooth_top(r_mesh_0, p_private_0, p_halo_0, n_nodes, smoother, relax, sweeps)

    var residual : double = 0
    for color in p_private_0.colors do
      residual += compute_residual(p_halo_0[color], p_private_0[color], n_nodes)
    end
    residual = sqrt(residual)
    if residual0/residual > cmath.pow(10, tolerance) then
      converged = true
    end
    c.printf("iteration %d, residual %13e\n", num_iterations, residual)
    break
  end
  ---- c.printf("iterations: %d\n", num_iterations)
  --token = 0
  --for color in p_private_0.colors do
  --  token += block_task(p_private_0[color])
  --end
  --for color in p_private_1.colors do
  --  token += block_task(p_private_1[color])
  --end
  --wait_for(token)
  --var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("Total time: %.6f sec.\n", (ts_end - ts_start) * 1e-6)
end

regentlib.start(main)
