use foldhash::{HashSet, HashSetExt};
use std::cmp::{max, min};
use std::num::NonZeroU64;

use serde::Serialize;

use crate::state::{IPartID, NodeID};

pub type Coord = i64;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Point {
    pub values: Vec<Coord>,
}

impl Point {
    pub fn new(values: Vec<Coord>) -> Self {
        Point { values }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rect {
    pub lo: Point,
    pub hi: Point,
}

impl Rect {
    pub fn new(lo: Point, hi: Point) -> Self {
        assert!(lo.dim() == hi.dim());
        Rect { lo, hi }
    }
    pub fn dim(&self) -> usize {
        self.lo.dim()
    }
    pub fn volume(&self) -> u64 {
        let mut vol = 1;
        for idx in 0..self.dim() {
            let lo = self.lo.values[idx];
            let hi = self.hi.values[idx];
            if hi < lo {
                vol = 0;
                break;
            } else {
                vol *= (hi - lo + 1) as u64;
            }
        }
        vol
    }
    pub fn is_empty(&self) -> bool {
        self.volume() == 0
    }
    pub fn union_bbox_point(&self, point: &Point) -> Self {
        assert!(self.dim() == point.dim());
        let lo: Vec<_> = self
            .lo
            .values
            .iter()
            .zip(point.values.iter())
            .map(|(a, b)| min(*a, *b))
            .collect();
        let hi: Vec<_> = self
            .hi
            .values
            .iter()
            .zip(point.values.iter())
            .map(|(a, b)| max(*a, *b))
            .collect();
        Rect::new(Point::new(lo), Point::new(hi))
    }
    pub fn union_bbox_rect(&self, rect: &Rect) -> Self {
        assert!(self.dim() == rect.dim());
        let lo: Vec<_> = self
            .lo
            .values
            .iter()
            .zip(rect.lo.values.iter())
            .map(|(a, b)| min(*a, *b))
            .collect();
        let hi: Vec<_> = self
            .hi
            .values
            .iter()
            .zip(rect.hi.values.iter())
            .map(|(a, b)| max(*a, *b))
            .collect();
        Rect::new(Point::new(lo), Point::new(hi))
    }
    pub fn union_bbox_rect_in_place(&mut self, rect: &Rect) {
        assert!(self.dim() == rect.dim());
        self.lo
            .values
            .iter_mut()
            .zip(&rect.lo.values)
            .for_each(|(a, &b)| *a = min(*a, b));
        self.hi
            .values
            .iter_mut()
            .zip(&rect.hi.values)
            .for_each(|(a, &b)| *a = max(*a, b));
    }
    pub fn contains_point(&self, point: &Point) -> bool {
        assert!(point.dim() == self.dim());
        for idx in 0..point.dim() {
            if point.values[idx] < self.lo.values[idx] {
                return false;
            }
            if point.values[idx] > self.hi.values[idx] {
                return false;
            }
        }
        return true;
    }
    pub fn overlaps(&self, rect: &Rect) -> bool {
        assert!(rect.dim() == self.dim());
        for idx in 0..rect.dim() {
            if self.lo.values[idx] > self.hi.values[idx]
                || self.lo.values[idx] > rect.hi.values[idx]
                || rect.lo.values[idx] > self.hi.values[idx]
                || rect.lo.values[idx] > rect.hi.values[idx]
            {
                return false;
            }
        }
        return true;
    }
    pub fn intersection(&self, rect: &Rect) -> Rect {
        assert!(rect.dim() == self.dim());
        let lo: Vec<_> = self
            .lo
            .values
            .iter()
            .zip(rect.lo.values.iter())
            .map(|(a, b)| max(*a, *b))
            .collect();
        let hi: Vec<_> = self
            .hi
            .values
            .iter()
            .zip(rect.hi.values.iter())
            .map(|(a, b)| min(*a, *b))
            .collect();
        Rect::new(Point::new(lo), Point::new(hi))
    }
    pub fn dominates(&self, rect: &Rect) -> bool {
        if rect.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        for idx in 0..self.dim() {
            if rect.lo.values[idx] < self.lo.values[idx]
                || self.hi.values[idx] < rect.hi.values[idx]
            {
                return false;
            }
        }
        return true;
    }
    pub fn subtract(&self, rect: &Rect) -> Option<Vec<Rect>> {
        assert!(self.dim() == rect.dim());
        if !self.overlaps(rect) {
            return None;
        }
        let mut subrects = Vec::new();
        if !rect.dominates(self) {
            Rect::subtract_helper(self.clone(), rect, 0, &mut subrects);
        }
        Some(subrects)
    }
    fn subtract_helper(rect: Rect, other: &Rect, dim: usize, subrects: &mut Vec<Rect>) {
        if dim == rect.dim() {
            // Base case
            if rect.overlaps(other) {
                assert!(other.dominates(&rect));
            } else {
                subrects.push(rect.clone());
            }
        } else {
            // Recursive case through the dimensions
            // Figure out how to break this rectangle along this dimension
            if other.lo.values[dim] <= rect.lo.values[dim] {
                if other.hi.values[dim] < rect.hi.values[dim] {
                    // Dominate lower edge, two outputs
                    let mut hi = rect.hi.clone();
                    hi.values[dim] = other.hi.values[dim];
                    Rect::subtract_helper(Rect::new(rect.lo.clone(), hi), other, dim + 1, subrects);
                    let mut lo = rect.lo.clone();
                    lo.values[dim] = other.hi.values[dim] + 1;
                    Rect::subtract_helper(Rect::new(lo, rect.hi.clone()), other, dim + 1, subrects);
                } else {
                    // Dominate both edges, one output
                    Rect::subtract_helper(rect, other, dim + 1, subrects);
                }
            } else if other.hi.values[dim] >= rect.hi.values[dim] {
                // Dominate upper edge, two outputs
                let mut hi = rect.hi.clone();
                hi.values[dim] = other.lo.values[dim] - 1;
                Rect::subtract_helper(Rect::new(rect.lo.clone(), hi), other, dim + 1, subrects);
                let mut lo = rect.lo.clone();
                lo.values[dim] = other.lo.values[dim];
                Rect::subtract_helper(Rect::new(lo, rect.hi.clone()), other, dim + 1, subrects);
            } else {
                // No domination, three outputs
                let mut hi = rect.hi.clone();
                hi.values[dim] = other.lo.values[dim] - 1;
                Rect::subtract_helper(Rect::new(rect.lo.clone(), hi), other, dim + 1, subrects);
                let mut lo = rect.lo.clone();
                hi = rect.hi.clone();
                lo.values[dim] = other.lo.values[dim];
                hi.values[dim] = other.hi.values[dim];
                Rect::subtract_helper(Rect::new(lo, hi), other, dim + 1, subrects);
                lo = rect.lo.clone();
                lo.values[dim] = other.hi.values[dim] + 1;
                Rect::subtract_helper(Rect::new(lo, rect.hi.clone()), other, dim + 1, subrects);
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Bounds {
    Point(Point),
    Rect(Rect),
    Empty,
    Unknown,
}

#[derive(Debug)]
pub struct ISpaceSize {
    pub dense_size: u64,
    pub sparse_size: u64,
    pub is_sparse: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ISpaceID(pub NonZeroU64);

#[derive(Debug)]
pub struct ISpace {
    pub ispace_id: ISpaceID,
    pub bounds: Bounds,
    pub points: Vec<Bounds>,
    pub name: Option<String>,
    pub parent: Option<IPartID>,
    pub size: Option<ISpaceSize>,
    first_node: Option<NodeID>,
}

impl ISpace {
    pub fn new(ispace_id: ISpaceID) -> Self {
        ISpace {
            ispace_id,
            bounds: Bounds::Unknown,
            points: Vec::new(),
            name: None,
            parent: None,
            size: None,
            first_node: None,
        }
    }

    // Important: these methods can get called multiple times in a
    // sparse instance. In this case the bounds will NOT be
    // accurate. But we don't use bounds in such cases anyway since we
    // refer to the dense/sparse sizes.
    pub fn set_point(&mut self, dim: u32, values: &[Coord], node: NodeID) -> &mut Self {
        // Only need to record geometry from one node because we trust that Realm has done its job
        if let Some(first) = self.first_node {
            if first != node {
                return self;
            }
        } else {
            self.first_node = Some(node);
        }
        let new_point = Point::new(values[0..(dim as usize)].to_owned());
        // Update the bounds if necessary
        self.bounds = match &self.bounds {
            Bounds::Rect(rect) => Bounds::Rect(rect.union_bbox_point(&new_point)),
            Bounds::Unknown => Bounds::Rect(Rect::new(new_point.clone(), new_point.clone())),
            _ => {
                panic!("Bounds should be a rectangle or unknown");
            }
        };
        self.points.push(Bounds::Point(new_point));
        self
    }
    pub fn set_rect(
        &mut self,
        dim: u32,
        values: &[Coord],
        max_dim: i32,
        node: NodeID,
    ) -> &mut Self {
        // Only need to record geometry from one node because we trust that Realm has done its job
        if let Some(first) = self.first_node {
            if first != node {
                return self;
            }
        } else {
            self.first_node = Some(node);
        }
        let new_rect = Rect::new(
            Point::new(values[0..(dim as usize)].to_owned()),
            Point::new(values[(max_dim as usize)..(max_dim as usize) + (dim as usize)].to_owned()),
        );
        // Update the bounds if necessary
        self.bounds = match &self.bounds {
            Bounds::Rect(rect) => Bounds::Rect(rect.union_bbox_rect(&new_rect)),
            Bounds::Unknown => Bounds::Rect(new_rect.clone()),
            _ => {
                panic!("Bounds should be a rectangle or unknown");
            }
        };
        self.points.push(Bounds::Rect(new_rect));
        self
    }
    pub fn set_empty(&mut self) -> &mut Self {
        assert!(self.bounds == Bounds::Unknown || self.bounds == Bounds::Empty);
        self.bounds = Bounds::Empty;
        self
    }
    pub fn set_name(&mut self, name: &str) -> &mut Self {
        assert!(self.name.as_ref().is_none_or(|x| x == name));
        self.name = Some(name.to_owned());
        self
    }
    pub fn set_parent(&mut self, parent: IPartID) -> &mut Self {
        assert!(self.parent.is_none());
        self.parent = Some(parent);
        self
    }
    pub fn set_size(&mut self, dense_size: u64, sparse_size: u64, is_sparse: bool) -> &mut Self {
        if let Some(space_size) = &self.size {
            assert!(space_size.dense_size == dense_size);
            assert!(space_size.sparse_size == sparse_size);
            assert!(space_size.is_sparse == is_sparse);
        } else {
            self.size = Some(ISpaceSize {
                dense_size,
                sparse_size,
                is_sparse,
            });
        }
        self
    }
    pub fn is_empty(&self) -> bool {
        match self.bounds {
            Bounds::Rect(_) => {
                assert!(!self.points.is_empty());
                false
            }
            Bounds::Empty => {
                assert!(self.points.is_empty());
                true
            }
            _ => {
                panic!("Unknown bounds for {:?}", self.ispace_id);
            }
        }
    }
    pub fn is_sparse(&self) -> bool {
        match self.bounds {
            Bounds::Rect(_) => {
                assert!(!self.points.is_empty());
                self.points.len() > 1
            }
            Bounds::Empty => {
                assert!(self.points.is_empty());
                false
            }
            _ => {
                panic!("Unknown bounds for {:?}", self.ispace_id);
            }
        }
    }
    pub fn volume(&self) -> u64 {
        let mut result = 0;
        for entry in &self.points {
            match entry {
                Bounds::Rect(rect) => {
                    result += rect.volume();
                }
                Bounds::Point(_) => {
                    result += 1;
                }
                _ => {
                    panic!("Bad bounds entry in {:?}", self.ispace_id);
                }
            }
        }
        result
    }
    pub fn sparsity_percentage(&self) -> f64 {
        let total_points = self.volume();
        assert!(total_points > 0);
        match &self.bounds {
            Bounds::Rect(rect) => {
                let bounds_volume = rect.volume();
                assert!(total_points <= bounds_volume);
                let numerator = 100.0 * (total_points as f64);
                let denominator = bounds_volume as f64;
                numerator / denominator
            }
            _ => {
                panic!("Bad index space bounds in {:?}", self.ispace_id);
            }
        }
    }
    pub fn contains_point(&self, point: &Point) -> bool {
        match &self.bounds {
            Bounds::Rect(rect) => {
                if !rect.contains_point(point) {
                    return false;
                }
            }
            _ => unreachable!(),
        }
        for entry in &self.points {
            match entry {
                Bounds::Point(other) => {
                    if other == point {
                        return true;
                    }
                }
                Bounds::Rect(rect) => {
                    if rect.contains_point(point) {
                        return true;
                    }
                }
                _ => {
                    unreachable!();
                }
            }
        }
        false
    }
    pub fn overlaps(&self, rect: &Rect) -> bool {
        match &self.bounds {
            Bounds::Rect(other) => {
                if !other.overlaps(rect) {
                    return false;
                }
            }
            _ => {
                unreachable!();
            }
        }
        for entry in &self.points {
            match entry {
                Bounds::Point(point) => {
                    if rect.contains_point(point) {
                        return true;
                    }
                }
                Bounds::Rect(other) => {
                    if rect.overlaps(other) {
                        return true;
                    }
                }
                _ => {
                    unreachable!();
                }
            }
        }
        false
    }
}

// R-tree spatial index for accelerating rectangle overlap queries.
// Uses Sort-Tile-Recursive (STR) bulk loading.
const RTREE_BRANCH_FACTOR: usize = 16;
const RTREE_THRESHOLD: usize = 64;

struct RTreeNode {
    bbox: Rect,
    children: RTreeChildren,
}

enum RTreeChildren {
    Leaves(Vec<usize>), // indices into the original rect slice
    Internal(Vec<RTreeNode>),
}

struct RTree<'a> {
    rects: &'a [Rect],
    root: Option<RTreeNode>,
}

impl<'a> RTree<'a> {
    fn build(rects: &'a [Rect]) -> Self {
        if rects.is_empty() {
            return RTree { rects, root: None };
        }
        let indices: Vec<usize> = (0..rects.len()).collect();
        let root = Self::build_node(rects, indices, 0);
        RTree {
            rects,
            root: Some(root),
        }
    }

    fn build_node(rects: &[Rect], mut indices: Vec<usize>, dim: usize) -> RTreeNode {
        if indices.len() <= RTREE_BRANCH_FACTOR {
            // Leaf node
            let bbox = Self::compute_bbox(rects, &indices);
            RTreeNode {
                bbox,
                children: RTreeChildren::Leaves(indices),
            }
        } else {
            let ndim = rects[indices[0]].dim();
            let sort_dim = dim % ndim;
            // Sort by center coordinate along current dimension
            indices.sort_by(|&a, &b| {
                let ca = rects[a].lo.values[sort_dim] + rects[a].hi.values[sort_dim];
                let cb = rects[b].lo.values[sort_dim] + rects[b].hi.values[sort_dim];
                ca.cmp(&cb)
            });
            // Partition into groups and recurse on next dimension
            let num_slices = (indices.len() + RTREE_BRANCH_FACTOR - 1) / RTREE_BRANCH_FACTOR;
            let slice_size = (indices.len() + num_slices - 1) / num_slices;
            let mut child_nodes = Vec::with_capacity(num_slices);
            for chunk in indices.chunks(slice_size) {
                child_nodes.push(Self::build_node(rects, chunk.to_vec(), dim + 1));
            }
            let bbox = Self::compute_bbox_from_nodes(&child_nodes);
            RTreeNode {
                bbox,
                children: RTreeChildren::Internal(child_nodes),
            }
        }
    }

    fn compute_bbox(rects: &[Rect], indices: &[usize]) -> Rect {
        indices
            .iter()
            .skip(1)
            .fold(rects[indices[0]].clone(), |mut acc, &idx| {
                acc.union_bbox_rect_in_place(&rects[idx]);
                acc
            })
    }

    fn compute_bbox_from_nodes(nodes: &[RTreeNode]) -> Rect {
        nodes
            .iter()
            .skip(1)
            .fold(nodes[0].bbox.clone(), |mut acc, n| {
                acc.union_bbox_rect_in_place(&n.bbox);
                acc
            })
    }

    fn query_overlaps(&self, query: &Rect, results: &mut Vec<usize>) {
        if let Some(ref root) = self.root {
            Self::query_node(self.rects, root, query, results);
        }
    }

    fn query_node(rects: &[Rect], node: &RTreeNode, query: &Rect, results: &mut Vec<usize>) {
        if !node.bbox.overlaps(query) {
            return;
        }
        match &node.children {
            RTreeChildren::Leaves(indices) => {
                for &idx in indices {
                    if rects[idx].overlaps(query) {
                        results.push(idx);
                    }
                }
            }
            RTreeChildren::Internal(children) => {
                for child in children {
                    Self::query_node(rects, child, query, results);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EquivalenceSet {
    pub rects: Vec<Rect>,
    pub spaces: HashSet<ISpaceID>,
}

impl EquivalenceSet {
    pub fn new(space: &ISpace) -> Self {
        let mut rects = Vec::new();
        for point in &space.points {
            match point {
                Bounds::Rect(rect) => rects.push(rect.clone()),
                Bounds::Point(point) => rects.push(Rect::new(point.clone(), point.clone())),
                _ => panic!("Bad bounds entry"),
            }
        }
        let mut spaces = HashSet::new();
        spaces.insert(space.ispace_id);
        EquivalenceSet { rects, spaces }
    }
    pub fn overlaps(&self, other: &EquivalenceSet) -> Option<EquivalenceSet> {
        let mut rects = Vec::new();
        if self.rects.len() >= RTREE_THRESHOLD || other.rects.len() >= RTREE_THRESHOLD {
            // Use R-tree acceleration
            let tree = RTree::build(&other.rects);
            let mut hits = Vec::new();
            for r1 in &self.rects {
                hits.clear();
                tree.query_overlaps(r1, &mut hits);
                for &idx in &hits {
                    let intersect = r1.intersection(&other.rects[idx]);
                    if !intersect.is_empty() {
                        rects.push(intersect);
                    }
                }
            }
        } else {
            // Brute force for small sets
            for r1 in &self.rects {
                for r2 in &other.rects {
                    let intersect = r1.intersection(r2);
                    if !intersect.is_empty() {
                        rects.push(intersect);
                    }
                }
            }
        }
        if !rects.is_empty() {
            Some(EquivalenceSet {
                rects,
                spaces: self.spaces.union(&other.spaces).cloned().collect(),
            })
        } else {
            None
        }
    }
    pub fn volume(&self) -> u64 {
        let mut result = 0;
        for rect in &self.rects {
            result += rect.volume();
        }
        result
    }
    pub fn is_empty(&self) -> bool {
        self.volume() == 0
    }
    pub fn subtract(&mut self, other: &EquivalenceSet) {
        if self.rects.len() >= RTREE_THRESHOLD || other.rects.len() >= RTREE_THRESHOLD {
            // Use R-tree acceleration: build tree on other, iterate self
            let tree = RTree::build(&other.rects);
            let mut new_rects = Vec::new();
            let mut hits = Vec::new();
            let old_rects = std::mem::take(&mut self.rects);
            for rect in old_rects {
                hits.clear();
                tree.query_overlaps(&rect, &mut hits);
                if hits.is_empty() {
                    new_rects.push(rect);
                } else {
                    // Subtract all overlapping rects from other
                    let mut pending = vec![rect];
                    for &idx in &hits {
                        let orect = &other.rects[idx];
                        let mut next_pending = Vec::new();
                        for r in pending {
                            if let Some(mut subrects) = r.subtract(orect) {
                                next_pending.append(&mut subrects);
                            } else {
                                next_pending.push(r);
                            }
                        }
                        pending = next_pending;
                    }
                    new_rects.extend(pending);
                }
            }
            self.rects = new_rects;
        } else {
            // Brute force for small sets
            for orect in &other.rects {
                let mut old_rects = Vec::new();
                std::mem::swap(&mut old_rects, &mut self.rects);
                for rect in old_rects {
                    if let Some(mut subrects) = rect.subtract(orect) {
                        self.rects.append(&mut subrects);
                    } else {
                        self.rects.push(rect);
                    }
                }
            }
        }
    }
    pub fn clear(&mut self) {
        self.rects.clear();
        self.spaces.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions to create Points and Rects more easily
    fn point1d(x: i64) -> Point {
        Point::new(vec![x])
    }

    fn point2d(x: i64, y: i64) -> Point {
        Point::new(vec![x, y])
    }

    fn point3d(x: i64, y: i64, z: i64) -> Point {
        Point::new(vec![x, y, z])
    }

    fn rect1d(lo: i64, hi: i64) -> Rect {
        Rect::new(point1d(lo), point1d(hi))
    }

    fn rect2d(lo_x: i64, lo_y: i64, hi_x: i64, hi_y: i64) -> Rect {
        Rect::new(point2d(lo_x, lo_y), point2d(hi_x, hi_y))
    }

    fn rect3d(lo_x: i64, lo_y: i64, lo_z: i64, hi_x: i64, hi_y: i64, hi_z: i64) -> Rect {
        Rect::new(point3d(lo_x, lo_y, lo_z), point3d(hi_x, hi_y, hi_z))
    }

    // ==================== volume tests ====================

    mod volume_tests {
        use super::*;

        #[test]
        fn test_volume_1d_single_point() {
            // A single point has volume 1
            let r = rect1d(5, 5);
            assert_eq!(r.volume(), 1);
        }

        #[test]
        fn test_volume_1d_range() {
            // [0, 9] has 10 elements
            let r = rect1d(0, 9);
            assert_eq!(r.volume(), 10);
        }

        #[test]
        fn test_volume_1d_negative_coords() {
            // [-5, 4] has 10 elements
            let r = rect1d(-5, 4);
            assert_eq!(r.volume(), 10);
        }

        #[test]
        fn test_volume_1d_empty() {
            // hi < lo means empty rectangle
            let r = rect1d(5, 3);
            assert_eq!(r.volume(), 0);
        }

        #[test]
        fn test_volume_2d_unit_square() {
            // Single point in 2D
            let r = rect2d(0, 0, 0, 0);
            assert_eq!(r.volume(), 1);
        }

        #[test]
        fn test_volume_2d_square() {
            // [0,0] to [2,2] is a 3x3 square = 9 points
            let r = rect2d(0, 0, 2, 2);
            assert_eq!(r.volume(), 9);
        }

        #[test]
        fn test_volume_2d_rectangle() {
            // [0,0] to [3,1] is 4x2 = 8 points
            let r = rect2d(0, 0, 3, 1);
            assert_eq!(r.volume(), 8);
        }

        #[test]
        fn test_volume_2d_empty_x() {
            // Empty in x dimension
            let r = rect2d(5, 0, 3, 2);
            assert_eq!(r.volume(), 0);
        }

        #[test]
        fn test_volume_2d_empty_y() {
            // Empty in y dimension
            let r = rect2d(0, 5, 3, 2);
            assert_eq!(r.volume(), 0);
        }

        #[test]
        fn test_volume_3d_unit_cube() {
            // Single point in 3D
            let r = rect3d(1, 1, 1, 1, 1, 1);
            assert_eq!(r.volume(), 1);
        }

        #[test]
        fn test_volume_3d_cube() {
            // [0,0,0] to [2,2,2] is 3x3x3 = 27 points
            let r = rect3d(0, 0, 0, 2, 2, 2);
            assert_eq!(r.volume(), 27);
        }

        #[test]
        fn test_volume_3d_cuboid() {
            // [0,0,0] to [3,1,4] is 4x2x5 = 40 points
            let r = rect3d(0, 0, 0, 3, 1, 4);
            assert_eq!(r.volume(), 40);
        }

        #[test]
        fn test_volume_3d_empty() {
            // Empty in z dimension
            let r = rect3d(0, 0, 5, 3, 3, 2);
            assert_eq!(r.volume(), 0);
        }
    }

    // ==================== union_bbox_point tests ====================

    mod union_bbox_point_tests {
        use super::*;

        #[test]
        fn test_union_bbox_point_1d_inside() {
            // Point inside rect - no change
            let r = rect1d(0, 10);
            let p = point1d(5);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![10]);
        }

        #[test]
        fn test_union_bbox_point_1d_below() {
            // Point below rect - extends lo
            let r = rect1d(5, 10);
            let p = point1d(2);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![2]);
            assert_eq!(result.hi.values, vec![10]);
        }

        #[test]
        fn test_union_bbox_point_1d_above() {
            // Point above rect - extends hi
            let r = rect1d(0, 5);
            let p = point1d(10);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![10]);
        }

        #[test]
        fn test_union_bbox_point_2d_inside() {
            let r = rect2d(0, 0, 10, 10);
            let p = point2d(5, 5);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0, 0]);
            assert_eq!(result.hi.values, vec![10, 10]);
        }

        #[test]
        fn test_union_bbox_point_2d_outside_corner() {
            // Point outside both dimensions
            let r = rect2d(0, 0, 5, 5);
            let p = point2d(10, 10);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0, 0]);
            assert_eq!(result.hi.values, vec![10, 10]);
        }

        #[test]
        fn test_union_bbox_point_2d_outside_negative() {
            // Point with negative coords
            let r = rect2d(0, 0, 5, 5);
            let p = point2d(-3, -7);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![-3, -7]);
            assert_eq!(result.hi.values, vec![5, 5]);
        }

        #[test]
        fn test_union_bbox_point_3d_inside() {
            let r = rect3d(0, 0, 0, 10, 10, 10);
            let p = point3d(5, 5, 5);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0, 0, 0]);
            assert_eq!(result.hi.values, vec![10, 10, 10]);
        }

        #[test]
        fn test_union_bbox_point_3d_outside() {
            let r = rect3d(0, 0, 0, 5, 5, 5);
            let p = point3d(10, -2, 8);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0, -2, 0]);
            assert_eq!(result.hi.values, vec![10, 5, 8]);
        }

        #[test]
        fn test_union_bbox_point_3d_on_boundary() {
            let r = rect3d(0, 0, 0, 5, 5, 5);
            let p = point3d(0, 5, 3);
            let result = r.union_bbox_point(&p);
            assert_eq!(result.lo.values, vec![0, 0, 0]);
            assert_eq!(result.hi.values, vec![5, 5, 5]);
        }
    }

    // ==================== union_bbox_rect tests ====================

    mod union_bbox_rect_tests {
        use super::*;

        #[test]
        fn test_union_bbox_rect_1d_disjoint() {
            let r1 = rect1d(0, 5);
            let r2 = rect1d(10, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![15]);
        }

        #[test]
        fn test_union_bbox_rect_1d_overlapping() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(5, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![15]);
        }

        #[test]
        fn test_union_bbox_rect_1d_contained() {
            // r2 is inside r1
            let r1 = rect1d(0, 20);
            let r2 = rect1d(5, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![20]);
        }

        #[test]
        fn test_union_bbox_rect_1d_same() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(0, 10);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0]);
            assert_eq!(result.hi.values, vec![10]);
        }

        #[test]
        fn test_union_bbox_rect_2d_disjoint() {
            let r1 = rect2d(0, 0, 5, 5);
            let r2 = rect2d(10, 10, 15, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0]);
            assert_eq!(result.hi.values, vec![15, 15]);
        }

        #[test]
        fn test_union_bbox_rect_2d_overlapping() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(5, 5, 15, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0]);
            assert_eq!(result.hi.values, vec![15, 15]);
        }

        #[test]
        fn test_union_bbox_rect_2d_adjacent() {
            // Touching but not overlapping
            let r1 = rect2d(0, 0, 5, 5);
            let r2 = rect2d(6, 0, 10, 5);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0]);
            assert_eq!(result.hi.values, vec![10, 5]);
        }

        #[test]
        fn test_union_bbox_rect_2d_negative_coords() {
            let r1 = rect2d(-10, -10, -5, -5);
            let r2 = rect2d(5, 5, 10, 10);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![-10, -10]);
            assert_eq!(result.hi.values, vec![10, 10]);
        }

        #[test]
        fn test_union_bbox_rect_3d_disjoint() {
            let r1 = rect3d(0, 0, 0, 5, 5, 5);
            let r2 = rect3d(10, 10, 10, 15, 15, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0, 0]);
            assert_eq!(result.hi.values, vec![15, 15, 15]);
        }

        #[test]
        fn test_union_bbox_rect_3d_overlapping() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0, 0]);
            assert_eq!(result.hi.values, vec![15, 15, 15]);
        }

        #[test]
        fn test_union_bbox_rect_3d_contained() {
            let r1 = rect3d(0, 0, 0, 20, 20, 20);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            let result = r1.union_bbox_rect(&r2);
            assert_eq!(result.lo.values, vec![0, 0, 0]);
            assert_eq!(result.hi.values, vec![20, 20, 20]);
        }
    }

    // ==================== contains_point tests ====================

    mod contains_point_tests {
        use super::*;

        #[test]
        fn test_contains_point_1d_inside() {
            let r = rect1d(0, 10);
            assert!(r.contains_point(&point1d(5)));
        }

        #[test]
        fn test_contains_point_1d_on_lo_boundary() {
            let r = rect1d(0, 10);
            assert!(r.contains_point(&point1d(0)));
        }

        #[test]
        fn test_contains_point_1d_on_hi_boundary() {
            let r = rect1d(0, 10);
            assert!(r.contains_point(&point1d(10)));
        }

        #[test]
        fn test_contains_point_1d_below() {
            let r = rect1d(0, 10);
            assert!(!r.contains_point(&point1d(-1)));
        }

        #[test]
        fn test_contains_point_1d_above() {
            let r = rect1d(0, 10);
            assert!(!r.contains_point(&point1d(11)));
        }

        #[test]
        fn test_contains_point_2d_inside() {
            let r = rect2d(0, 0, 10, 10);
            assert!(r.contains_point(&point2d(5, 5)));
        }

        #[test]
        fn test_contains_point_2d_on_corner() {
            let r = rect2d(0, 0, 10, 10);
            assert!(r.contains_point(&point2d(0, 0)));
            assert!(r.contains_point(&point2d(10, 10)));
            assert!(r.contains_point(&point2d(0, 10)));
            assert!(r.contains_point(&point2d(10, 0)));
        }

        #[test]
        fn test_contains_point_2d_on_edge() {
            let r = rect2d(0, 0, 10, 10);
            assert!(r.contains_point(&point2d(5, 0)));
            assert!(r.contains_point(&point2d(0, 5)));
            assert!(r.contains_point(&point2d(10, 5)));
            assert!(r.contains_point(&point2d(5, 10)));
        }

        #[test]
        fn test_contains_point_2d_outside_x() {
            let r = rect2d(0, 0, 10, 10);
            assert!(!r.contains_point(&point2d(-1, 5)));
            assert!(!r.contains_point(&point2d(11, 5)));
        }

        #[test]
        fn test_contains_point_2d_outside_y() {
            let r = rect2d(0, 0, 10, 10);
            assert!(!r.contains_point(&point2d(5, -1)));
            assert!(!r.contains_point(&point2d(5, 11)));
        }

        #[test]
        fn test_contains_point_2d_outside_both() {
            let r = rect2d(0, 0, 10, 10);
            assert!(!r.contains_point(&point2d(-1, -1)));
            assert!(!r.contains_point(&point2d(11, 11)));
        }

        #[test]
        fn test_contains_point_3d_inside() {
            let r = rect3d(0, 0, 0, 10, 10, 10);
            assert!(r.contains_point(&point3d(5, 5, 5)));
        }

        #[test]
        fn test_contains_point_3d_on_corner() {
            let r = rect3d(0, 0, 0, 10, 10, 10);
            assert!(r.contains_point(&point3d(0, 0, 0)));
            assert!(r.contains_point(&point3d(10, 10, 10)));
            assert!(r.contains_point(&point3d(0, 0, 10)));
            assert!(r.contains_point(&point3d(10, 10, 0)));
        }

        #[test]
        fn test_contains_point_3d_outside() {
            let r = rect3d(0, 0, 0, 10, 10, 10);
            assert!(!r.contains_point(&point3d(-1, 5, 5)));
            assert!(!r.contains_point(&point3d(5, -1, 5)));
            assert!(!r.contains_point(&point3d(5, 5, -1)));
            assert!(!r.contains_point(&point3d(11, 5, 5)));
            assert!(!r.contains_point(&point3d(5, 11, 5)));
            assert!(!r.contains_point(&point3d(5, 5, 11)));
        }

        #[test]
        fn test_contains_point_3d_negative_coords() {
            let r = rect3d(-10, -10, -10, 10, 10, 10);
            assert!(r.contains_point(&point3d(0, 0, 0)));
            assert!(r.contains_point(&point3d(-5, -5, -5)));
            assert!(!r.contains_point(&point3d(-11, 0, 0)));
        }
    }

    // ==================== overlaps tests ====================

    mod overlaps_tests {
        use super::*;

        #[test]
        fn test_overlaps_1d_same() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(0, 10);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_1d_partial() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(5, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_1d_touching() {
            // Sharing a single point
            let r1 = rect1d(0, 10);
            let r2 = rect1d(10, 20);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_1d_disjoint() {
            let r1 = rect1d(0, 5);
            let r2 = rect1d(10, 15);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_1d_contained() {
            let r1 = rect1d(0, 20);
            let r2 = rect1d(5, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_1d_empty_rect() {
            // Empty rectangle (hi < lo) should not overlap
            let r1 = rect1d(5, 3); // empty
            let r2 = rect1d(0, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_same() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(0, 0, 10, 10);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_partial() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(5, 5, 15, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_touching_corner() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(10, 10, 20, 20);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_touching_edge() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(10, 0, 20, 10);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_disjoint_x() {
            let r1 = rect2d(0, 0, 5, 10);
            let r2 = rect2d(10, 0, 15, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_disjoint_y() {
            let r1 = rect2d(0, 0, 10, 5);
            let r2 = rect2d(0, 10, 10, 15);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_disjoint_both() {
            let r1 = rect2d(0, 0, 5, 5);
            let r2 = rect2d(10, 10, 15, 15);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_contained() {
            let r1 = rect2d(0, 0, 20, 20);
            let r2 = rect2d(5, 5, 15, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_2d_empty_rect() {
            let r1 = rect2d(5, 0, 3, 10); // empty in x
            let r2 = rect2d(0, 0, 10, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_same() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(0, 0, 0, 10, 10, 10);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_partial() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_touching_corner() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(10, 10, 10, 20, 20, 20);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_disjoint_x() {
            let r1 = rect3d(0, 0, 0, 5, 10, 10);
            let r2 = rect3d(10, 0, 0, 15, 10, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_disjoint_y() {
            let r1 = rect3d(0, 0, 0, 10, 5, 10);
            let r2 = rect3d(0, 10, 0, 10, 15, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_disjoint_z() {
            let r1 = rect3d(0, 0, 0, 10, 10, 5);
            let r2 = rect3d(0, 0, 10, 10, 10, 15);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_contained() {
            let r1 = rect3d(0, 0, 0, 20, 20, 20);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            assert!(r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_empty_rect() {
            let r1 = rect3d(0, 0, 5, 10, 10, 3); // empty in z
            let r2 = rect3d(0, 0, 0, 10, 10, 10);
            assert!(!r1.overlaps(&r2));
        }

        #[test]
        fn test_overlaps_3d_negative_coords() {
            let r1 = rect3d(-10, -10, -10, 0, 0, 0);
            let r2 = rect3d(-5, -5, -5, 5, 5, 5);
            assert!(r1.overlaps(&r2));
        }
    }

    // ==================== intersection tests ====================

    mod intersection_tests {
        use super::*;

        #[test]
        fn test_intersection_1d_same() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(0, 10);
            assert_eq!(r1.intersection(&r2), rect1d(0, 10));
        }

        #[test]
        fn test_intersection_1d_partial() {
            let r1 = rect1d(0, 10);
            let r2 = rect1d(5, 15);
            assert_eq!(r1.intersection(&r2), rect1d(5, 10));
        }

        #[test]
        fn test_intersection_1d_contained() {
            // When r2 is fully inside r1, the intersection is r2.
            let r1 = rect1d(0, 20);
            let r2 = rect1d(5, 15);
            assert_eq!(r1.intersection(&r2), rect1d(5, 15));
        }

        #[test]
        fn test_intersection_1d_touching() {
            // Sharing a single point produces a unit-volume result.
            let r1 = rect1d(0, 10);
            let r2 = rect1d(10, 20);
            let result = r1.intersection(&r2);
            assert_eq!(result, rect1d(10, 10));
            assert_eq!(result.volume(), 1);
        }

        #[test]
        fn test_intersection_1d_disjoint() {
            // Disjoint inputs produce an empty (hi < lo) rectangle.
            let r1 = rect1d(0, 5);
            let r2 = rect1d(10, 15);
            let result = r1.intersection(&r2);
            assert!(result.is_empty());
        }

        #[test]
        fn test_intersection_1d_negative_coords() {
            let r1 = rect1d(-10, 5);
            let r2 = rect1d(-3, 10);
            assert_eq!(r1.intersection(&r2), rect1d(-3, 5));
        }

        #[test]
        fn test_intersection_1d_single_point_each() {
            let r1 = rect1d(7, 7);
            let r2 = rect1d(7, 7);
            assert_eq!(r1.intersection(&r2), rect1d(7, 7));
        }

        #[test]
        fn test_intersection_2d_same() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(0, 0, 10, 10);
            assert_eq!(r1.intersection(&r2), rect2d(0, 0, 10, 10));
        }

        #[test]
        fn test_intersection_2d_partial() {
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(5, 5, 15, 15);
            assert_eq!(r1.intersection(&r2), rect2d(5, 5, 10, 10));
        }

        #[test]
        fn test_intersection_2d_contained() {
            let r1 = rect2d(0, 0, 20, 20);
            let r2 = rect2d(5, 5, 15, 15);
            assert_eq!(r1.intersection(&r2), rect2d(5, 5, 15, 15));
        }

        #[test]
        fn test_intersection_2d_touching_corner() {
            // Two rects touching at a single corner intersect at that point.
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(10, 10, 20, 20);
            let result = r1.intersection(&r2);
            assert_eq!(result, rect2d(10, 10, 10, 10));
            assert_eq!(result.volume(), 1);
        }

        #[test]
        fn test_intersection_2d_touching_edge() {
            // Touching on a shared edge collapses to a 1xN rectangle.
            let r1 = rect2d(0, 0, 10, 10);
            let r2 = rect2d(10, 0, 20, 10);
            let result = r1.intersection(&r2);
            assert_eq!(result, rect2d(10, 0, 10, 10));
            assert_eq!(result.volume(), 11);
        }

        #[test]
        fn test_intersection_2d_disjoint_x() {
            let r1 = rect2d(0, 0, 5, 10);
            let r2 = rect2d(10, 0, 15, 10);
            let result = r1.intersection(&r2);
            assert!(result.is_empty());
        }

        #[test]
        fn test_intersection_2d_disjoint_y() {
            let r1 = rect2d(0, 0, 10, 5);
            let r2 = rect2d(0, 10, 10, 15);
            let result = r1.intersection(&r2);
            assert!(result.is_empty());
        }

        #[test]
        fn test_intersection_2d_disjoint_both() {
            let r1 = rect2d(0, 0, 5, 5);
            let r2 = rect2d(10, 10, 15, 15);
            let result = r1.intersection(&r2);
            assert!(result.is_empty());
        }

        #[test]
        fn test_intersection_2d_cross_shape() {
            // A horizontal and vertical strip intersect in their shared square.
            let horizontal = rect2d(-10, 0, 10, 2);
            let vertical = rect2d(0, -10, 2, 10);
            assert_eq!(horizontal.intersection(&vertical), rect2d(0, 0, 2, 2));
        }

        #[test]
        fn test_intersection_3d_same() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(0, 0, 0, 10, 10, 10);
            assert_eq!(r1.intersection(&r2), rect3d(0, 0, 0, 10, 10, 10));
        }

        #[test]
        fn test_intersection_3d_partial() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            assert_eq!(r1.intersection(&r2), rect3d(5, 5, 5, 10, 10, 10));
        }

        #[test]
        fn test_intersection_3d_contained() {
            let r1 = rect3d(0, 0, 0, 20, 20, 20);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            assert_eq!(r1.intersection(&r2), rect3d(5, 5, 5, 15, 15, 15));
        }

        #[test]
        fn test_intersection_3d_touching_corner() {
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(10, 10, 10, 20, 20, 20);
            let result = r1.intersection(&r2);
            assert_eq!(result, rect3d(10, 10, 10, 10, 10, 10));
            assert_eq!(result.volume(), 1);
        }

        #[test]
        fn test_intersection_3d_disjoint_z() {
            let r1 = rect3d(0, 0, 0, 10, 10, 5);
            let r2 = rect3d(0, 0, 10, 10, 10, 15);
            let result = r1.intersection(&r2);
            assert!(result.is_empty());
        }

        #[test]
        fn test_intersection_3d_negative_coords() {
            let r1 = rect3d(-10, -10, -10, 0, 0, 0);
            let r2 = rect3d(-5, -5, -5, 5, 5, 5);
            assert_eq!(r1.intersection(&r2), rect3d(-5, -5, -5, 0, 0, 0));
        }

        #[test]
        fn test_intersection_commutative() {
            // a.intersection(b) should equal b.intersection(a).
            let r1 = rect3d(0, 0, 0, 10, 10, 10);
            let r2 = rect3d(5, 5, 5, 15, 15, 15);
            assert_eq!(r1.intersection(&r2), r2.intersection(&r1));
        }

        #[test]
        fn test_intersection_idempotent() {
            // a.intersection(a) should equal a.
            let r = rect2d(3, -4, 12, 7);
            assert_eq!(r.intersection(&r), r);
        }

        #[test]
        fn test_intersection_volume_consistent_with_overlaps() {
            // Whenever overlaps() is true, the intersection must be non-empty;
            // whenever overlaps() is false, the intersection must be empty.
            let overlapping = rect2d(0, 0, 10, 10);
            let other = rect2d(5, 5, 15, 15);
            assert!(overlapping.overlaps(&other));
            assert!(!overlapping.intersection(&other).is_empty());

            let disjoint_a = rect2d(0, 0, 5, 5);
            let disjoint_b = rect2d(10, 10, 15, 15);
            assert!(!disjoint_a.overlaps(&disjoint_b));
            assert!(disjoint_a.intersection(&disjoint_b).is_empty());
        }

        #[test]
        #[should_panic]
        fn test_intersection_mismatched_dims_panics() {
            // The method asserts on matching dimensionality.
            let r1 = rect1d(0, 10);
            let r2 = rect2d(0, 0, 10, 10);
            let _ = r1.intersection(&r2);
        }
    }

    // ==================== RTree tests ====================

    mod rtree_tests {
        use super::*;

        // Reference implementation of query_overlaps for validating the tree.
        fn brute_force_overlaps(rects: &[Rect], query: &Rect) -> Vec<usize> {
            (0..rects.len())
                .filter(|&i| rects[i].overlaps(query))
                .collect()
        }

        fn query_sorted(tree: &RTree, query: &Rect) -> Vec<usize> {
            let mut results = Vec::new();
            tree.query_overlaps(query, &mut results);
            results.sort();
            results
        }

        #[test]
        fn test_rtree_build_empty() {
            let rects: Vec<Rect> = Vec::new();
            let tree = RTree::build(&rects);
            assert!(tree.root.is_none());
        }

        #[test]
        fn test_rtree_query_empty_returns_no_results() {
            let rects: Vec<Rect> = Vec::new();
            let tree = RTree::build(&rects);
            let mut results = Vec::new();
            tree.query_overlaps(&rect2d(0, 0, 10, 10), &mut results);
            assert!(results.is_empty());
        }

        #[test]
        fn test_rtree_single_rect_is_leaf_root() {
            let rects = vec![rect2d(0, 0, 5, 5)];
            let tree = RTree::build(&rects);
            let root = tree.root.as_ref().expect("expected a root node");
            // A single-rect tree: root is a leaf whose bbox equals the rect.
            assert_eq!(root.bbox, rect2d(0, 0, 5, 5));
            assert!(matches!(root.children, RTreeChildren::Leaves(_)));
        }

        #[test]
        fn test_rtree_leaf_root_for_small_input() {
            // <= RTREE_BRANCH_FACTOR rects fit directly in a leaf root.
            let rects: Vec<Rect> = (0..RTREE_BRANCH_FACTOR as i64)
                .map(|i| rect2d(i, 0, i + 1, 1))
                .collect();
            let tree = RTree::build(&rects);
            let root = tree.root.as_ref().unwrap();
            assert!(matches!(root.children, RTreeChildren::Leaves(_)));
            // The root bbox must dominate every input rect.
            for r in &rects {
                assert!(root.bbox.dominates(r));
            }
        }

        #[test]
        fn test_rtree_internal_root_for_large_input() {
            // > RTREE_BRANCH_FACTOR rects force an internal root.
            let rects: Vec<Rect> = (0..RTREE_BRANCH_FACTOR as i64 + 4)
                .map(|i| rect2d(i, i, i + 1, i + 1))
                .collect();
            let tree = RTree::build(&rects);
            let root = tree.root.as_ref().unwrap();
            assert!(matches!(root.children, RTreeChildren::Internal(_)));
            for r in &rects {
                assert!(root.bbox.dominates(r));
            }
        }

        #[test]
        fn test_rtree_query_finds_overlapping_rects() {
            let rects = vec![
                rect2d(0, 0, 5, 5),
                rect2d(10, 10, 15, 15),
                rect2d(4, 4, 12, 12),
            ];
            let tree = RTree::build(&rects);
            // The query overlaps rect 0 and rect 2 but not rect 1.
            assert_eq!(query_sorted(&tree, &rect2d(3, 3, 6, 6)), vec![0, 2]);
        }

        #[test]
        fn test_rtree_query_no_overlap() {
            let rects = vec![rect2d(0, 0, 5, 5), rect2d(10, 10, 15, 15)];
            let tree = RTree::build(&rects);
            let mut results = Vec::new();
            tree.query_overlaps(&rect2d(100, 100, 110, 110), &mut results);
            assert!(results.is_empty());
        }

        #[test]
        fn test_rtree_query_covers_all() {
            let rects: Vec<Rect> = (0..5).map(|i| rect2d(i * 10, 0, i * 10 + 5, 5)).collect();
            let tree = RTree::build(&rects);
            assert_eq!(
                query_sorted(&tree, &rect2d(-100, -100, 1000, 1000)),
                vec![0, 1, 2, 3, 4]
            );
        }

        #[test]
        fn test_rtree_query_touching_counts_as_overlap() {
            // Rect::overlaps treats touching as overlap; the tree must agree.
            let rects = vec![rect2d(0, 0, 10, 10)];
            let tree = RTree::build(&rects);
            assert_eq!(query_sorted(&tree, &rect2d(10, 10, 20, 20)), vec![0]);
        }

        #[test]
        fn test_rtree_matches_brute_force_2d() {
            // A 10x10 grid of rects (100 total) forces a multi-level tree;
            // every query result must match the brute-force answer.
            let mut rects = Vec::new();
            for i in 0..10 {
                for j in 0..10 {
                    rects.push(rect2d(i * 5, j * 5, i * 5 + 3, j * 5 + 3));
                }
            }
            let tree = RTree::build(&rects);
            let queries = vec![
                rect2d(0, 0, 10, 10),
                rect2d(-5, -5, 1, 1),
                rect2d(17, 17, 22, 22),
                rect2d(100, 100, 110, 110),
                rect2d(0, 0, 100, 100),
                rect2d(23, 2, 24, 48),
            ];
            for q in &queries {
                let actual = query_sorted(&tree, q);
                let mut expected = brute_force_overlaps(&rects, q);
                expected.sort();
                assert_eq!(actual, expected, "mismatch for query {:?}", q);
            }
        }

        #[test]
        fn test_rtree_matches_brute_force_3d() {
            // A 5x5x5 grid (125 rects) stresses STR sorting across all 3 dims.
            let mut rects = Vec::new();
            for i in 0..5 {
                for j in 0..5 {
                    for k in 0..5 {
                        rects.push(rect3d(i * 4, j * 4, k * 4, i * 4 + 2, j * 4 + 2, k * 4 + 2));
                    }
                }
            }
            let tree = RTree::build(&rects);
            let queries = vec![
                rect3d(0, 0, 0, 5, 5, 5),
                rect3d(-3, -3, -3, 0, 0, 0),
                rect3d(10, 10, 10, 14, 14, 14),
                rect3d(100, 100, 100, 200, 200, 200),
                rect3d(0, 0, 0, 100, 100, 100),
            ];
            for q in &queries {
                let actual = query_sorted(&tree, q);
                let mut expected = brute_force_overlaps(&rects, q);
                expected.sort();
                assert_eq!(actual, expected, "mismatch for query {:?}", q);
            }
        }

        #[test]
        fn test_rtree_no_duplicate_results() {
            // Every overlapping rect must appear exactly once in the results.
            let rects: Vec<Rect> = (0..50).map(|i| rect2d(i, 0, i + 1, 1)).collect();
            let tree = RTree::build(&rects);
            let mut results = Vec::new();
            tree.query_overlaps(&rect2d(-10, -10, 100, 100), &mut results);
            let len_before = results.len();
            results.sort();
            results.dedup();
            assert_eq!(results.len(), len_before);
            assert_eq!(results.len(), rects.len());
        }

        #[test]
        fn test_rtree_query_appends_to_results() {
            // query_overlaps should append to the caller's Vec, not clear it —
            // callers in EquivalenceSet rely on controlling when to clear.
            let rects = vec![rect2d(0, 0, 5, 5)];
            let tree = RTree::build(&rects);
            let mut results = vec![999];
            tree.query_overlaps(&rect2d(0, 0, 5, 5), &mut results);
            assert_eq!(results[0], 999);
            assert!(results.contains(&0));
        }

        #[test]
        fn test_rtree_handles_identical_rects() {
            // Duplicate input rects should each be indexed and returned.
            let rects: Vec<Rect> = (0..20).map(|_| rect2d(0, 0, 5, 5)).collect();
            let tree = RTree::build(&rects);
            let results = query_sorted(&tree, &rect2d(1, 1, 2, 2));
            assert_eq!(results, (0..20).collect::<Vec<usize>>());
        }

        #[test]
        fn test_rtree_1d_query() {
            // The tree must also work for 1D rects.
            let rects: Vec<Rect> = (0..30).map(|i| rect1d(i * 2, i * 2 + 1)).collect();
            let tree = RTree::build(&rects);
            let query = rect1d(5, 14);
            let actual = query_sorted(&tree, &query);
            let mut expected = brute_force_overlaps(&rects, &query);
            expected.sort();
            assert_eq!(actual, expected);
        }
    }

    // ==================== ISpace tests ====================

    mod ispace_tests {
        use super::*;

        // Helper to create a new ISpace with a given ID
        fn new_ispace(id: u64) -> ISpace {
            ISpace::new(ISpaceID(NonZeroU64::new(id).unwrap()))
        }

        // Helper to create an ISpace with a single rectangle
        fn ispace_with_rect_1d(id: u64, lo: i64, hi: i64) -> ISpace {
            let mut ispace = new_ispace(id);
            // set_rect takes: dim, values slice, max_dim
            // values slice format: [lo_coords..., hi_coords...]
            ispace.set_rect(1, &[lo, hi], 1, NodeID(0));
            ispace
        }

        fn ispace_with_rect_2d(id: u64, lo_x: i64, lo_y: i64, hi_x: i64, hi_y: i64) -> ISpace {
            let mut ispace = new_ispace(id);
            ispace.set_rect(2, &[lo_x, lo_y, hi_x, hi_y], 2, NodeID(0));
            ispace
        }

        fn ispace_with_rect_3d(
            id: u64,
            lo_x: i64,
            lo_y: i64,
            lo_z: i64,
            hi_x: i64,
            hi_y: i64,
            hi_z: i64,
        ) -> ISpace {
            let mut ispace = new_ispace(id);
            ispace.set_rect(3, &[lo_x, lo_y, lo_z, hi_x, hi_y, hi_z], 3, NodeID(0));
            ispace
        }

        // Helper to create an ISpace with a single point
        fn ispace_with_point_1d(id: u64, x: i64) -> ISpace {
            let mut ispace = new_ispace(id);
            ispace.set_point(1, &[x], NodeID(0));
            ispace
        }

        fn ispace_with_point_2d(id: u64, x: i64, y: i64) -> ISpace {
            let mut ispace = new_ispace(id);
            ispace.set_point(2, &[x, y], NodeID(0));
            ispace
        }

        fn ispace_with_point_3d(id: u64, x: i64, y: i64, z: i64) -> ISpace {
            let mut ispace = new_ispace(id);
            ispace.set_point(3, &[x, y, z], NodeID(0));
            ispace
        }

        // ==================== ISpace volume tests ====================

        mod ispace_volume_tests {
            use super::*;

            #[test]
            fn test_ispace_volume_1d_single_rect() {
                let ispace = ispace_with_rect_1d(1, 0, 9);
                assert_eq!(ispace.volume(), 10);
            }

            #[test]
            fn test_ispace_volume_1d_single_point() {
                let ispace = ispace_with_point_1d(1, 5);
                assert_eq!(ispace.volume(), 1);
            }

            #[test]
            fn test_ispace_volume_1d_multiple_rects() {
                let mut ispace = new_ispace(1);
                // Two disjoint rectangles: [0,4] and [10,14]
                ispace.set_rect(1, &[0, 4], 1, NodeID(0));
                ispace.set_rect(1, &[10, 14], 1, NodeID(0));
                // 5 + 5 = 10
                assert_eq!(ispace.volume(), 10);
            }

            #[test]
            fn test_ispace_volume_1d_multiple_points() {
                let mut ispace = new_ispace(1);
                ispace.set_point(1, &[0], NodeID(0));
                ispace.set_point(1, &[5], NodeID(0));
                ispace.set_point(1, &[10], NodeID(0));
                assert_eq!(ispace.volume(), 3);
            }

            #[test]
            fn test_ispace_volume_1d_mixed_rects_and_points() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(1, &[0, 4], 1, NodeID(0)); // 5 points
                ispace.set_point(1, &[10], NodeID(0)); // 1 point
                ispace.set_point(1, &[20], NodeID(0)); // 1 point
                assert_eq!(ispace.volume(), 7);
            }

            #[test]
            fn test_ispace_volume_2d_single_rect() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 3, 3);
                // 4x4 = 16
                assert_eq!(ispace.volume(), 16);
            }

            #[test]
            fn test_ispace_volume_2d_single_point() {
                let ispace = ispace_with_point_2d(1, 5, 5);
                assert_eq!(ispace.volume(), 1);
            }

            #[test]
            fn test_ispace_volume_2d_multiple_rects() {
                let mut ispace = new_ispace(1);
                // Two disjoint rectangles
                ispace.set_rect(2, &[0, 0, 2, 2], 2, NodeID(0)); // 3x3 = 9
                ispace.set_rect(2, &[10, 10, 12, 12], 2, NodeID(0)); // 3x3 = 9
                assert_eq!(ispace.volume(), 18);
            }

            #[test]
            fn test_ispace_volume_2d_mixed() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(2, &[0, 0, 1, 1], 2, NodeID(0)); // 2x2 = 4
                ispace.set_point(2, &[10, 10], NodeID(0)); // 1
                assert_eq!(ispace.volume(), 5);
            }

            #[test]
            fn test_ispace_volume_3d_single_rect() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 2, 2, 2);
                // 3x3x3 = 27
                assert_eq!(ispace.volume(), 27);
            }

            #[test]
            fn test_ispace_volume_3d_single_point() {
                let ispace = ispace_with_point_3d(1, 5, 5, 5);
                assert_eq!(ispace.volume(), 1);
            }

            #[test]
            fn test_ispace_volume_3d_multiple_rects() {
                let mut ispace = new_ispace(1);
                // Two disjoint cubes
                ispace.set_rect(3, &[0, 0, 0, 1, 1, 1], 3, NodeID(0)); // 2x2x2 = 8
                ispace.set_rect(3, &[10, 10, 10, 11, 11, 11], 3, NodeID(0)); // 2x2x2 = 8
                assert_eq!(ispace.volume(), 16);
            }

            #[test]
            fn test_ispace_volume_3d_mixed() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(3, &[0, 0, 0, 1, 1, 1], 3, NodeID(0)); // 2x2x2 = 8
                ispace.set_point(3, &[10, 10, 10], NodeID(0)); // 1
                ispace.set_point(3, &[20, 20, 20], NodeID(0)); // 1
                assert_eq!(ispace.volume(), 10);
            }
        }

        // ==================== ISpace contains_point tests ====================

        mod ispace_contains_point_tests {
            use super::*;

            #[test]
            fn test_ispace_contains_point_1d_in_rect() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(ispace.contains_point(&point1d(5)));
                assert!(ispace.contains_point(&point1d(0)));
                assert!(ispace.contains_point(&point1d(10)));
            }

            #[test]
            fn test_ispace_contains_point_1d_outside_rect() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(!ispace.contains_point(&point1d(-1)));
                assert!(!ispace.contains_point(&point1d(11)));
            }

            #[test]
            fn test_ispace_contains_point_1d_exact_point() {
                let ispace = ispace_with_point_1d(1, 5);
                assert!(ispace.contains_point(&point1d(5)));
                assert!(!ispace.contains_point(&point1d(4)));
                assert!(!ispace.contains_point(&point1d(6)));
            }

            #[test]
            fn test_ispace_contains_point_1d_multiple_rects() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(1, &[0, 4], 1, NodeID(0));
                ispace.set_rect(1, &[10, 14], 1, NodeID(0));
                // In first rect
                assert!(ispace.contains_point(&point1d(2)));
                // In second rect
                assert!(ispace.contains_point(&point1d(12)));
                // Between rects (in bounds but not in any rect)
                assert!(!ispace.contains_point(&point1d(7)));
            }

            #[test]
            fn test_ispace_contains_point_1d_multiple_points() {
                let mut ispace = new_ispace(1);
                ispace.set_point(1, &[0], NodeID(0));
                ispace.set_point(1, &[5], NodeID(0));
                ispace.set_point(1, &[10], NodeID(0));
                assert!(ispace.contains_point(&point1d(0)));
                assert!(ispace.contains_point(&point1d(5)));
                assert!(ispace.contains_point(&point1d(10)));
                assert!(!ispace.contains_point(&point1d(3)));
            }

            #[test]
            fn test_ispace_contains_point_2d_in_rect() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(ispace.contains_point(&point2d(5, 5)));
                assert!(ispace.contains_point(&point2d(0, 0)));
                assert!(ispace.contains_point(&point2d(10, 10)));
            }

            #[test]
            fn test_ispace_contains_point_2d_outside_rect() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(!ispace.contains_point(&point2d(-1, 5)));
                assert!(!ispace.contains_point(&point2d(5, -1)));
                assert!(!ispace.contains_point(&point2d(11, 5)));
                assert!(!ispace.contains_point(&point2d(5, 11)));
            }

            #[test]
            fn test_ispace_contains_point_2d_multiple_rects() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(2, &[0, 0, 4, 4], 2, NodeID(0));
                ispace.set_rect(2, &[10, 10, 14, 14], 2, NodeID(0));
                // In first rect
                assert!(ispace.contains_point(&point2d(2, 2)));
                // In second rect
                assert!(ispace.contains_point(&point2d(12, 12)));
                // In bounds but not in any rect
                assert!(!ispace.contains_point(&point2d(7, 7)));
            }

            #[test]
            fn test_ispace_contains_point_2d_mixed() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(2, &[0, 0, 4, 4], 2, NodeID(0));
                ispace.set_point(2, &[10, 10], NodeID(0));
                assert!(ispace.contains_point(&point2d(2, 2)));
                assert!(ispace.contains_point(&point2d(10, 10)));
                assert!(!ispace.contains_point(&point2d(7, 7)));
            }

            #[test]
            fn test_ispace_contains_point_3d_in_rect() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(ispace.contains_point(&point3d(5, 5, 5)));
                assert!(ispace.contains_point(&point3d(0, 0, 0)));
                assert!(ispace.contains_point(&point3d(10, 10, 10)));
            }

            #[test]
            fn test_ispace_contains_point_3d_outside_rect() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(!ispace.contains_point(&point3d(-1, 5, 5)));
                assert!(!ispace.contains_point(&point3d(5, -1, 5)));
                assert!(!ispace.contains_point(&point3d(5, 5, -1)));
                assert!(!ispace.contains_point(&point3d(11, 5, 5)));
            }

            #[test]
            fn test_ispace_contains_point_3d_multiple_rects() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(3, &[0, 0, 0, 4, 4, 4], 3, NodeID(0));
                ispace.set_rect(3, &[10, 10, 10, 14, 14, 14], 3, NodeID(0));
                // In first rect
                assert!(ispace.contains_point(&point3d(2, 2, 2)));
                // In second rect
                assert!(ispace.contains_point(&point3d(12, 12, 12)));
                // In bounds but not in any rect
                assert!(!ispace.contains_point(&point3d(7, 7, 7)));
            }

            #[test]
            fn test_ispace_contains_point_3d_mixed() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(3, &[0, 0, 0, 4, 4, 4], 3, NodeID(0));
                ispace.set_point(3, &[10, 10, 10], NodeID(0));
                assert!(ispace.contains_point(&point3d(2, 2, 2)));
                assert!(ispace.contains_point(&point3d(10, 10, 10)));
                assert!(!ispace.contains_point(&point3d(7, 7, 7)));
            }
        }

        // ==================== ISpace overlaps tests ====================

        mod ispace_overlaps_tests {
            use super::*;

            #[test]
            fn test_ispace_overlaps_1d_rect_inside() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(ispace.overlaps(&rect1d(2, 8)));
            }

            #[test]
            fn test_ispace_overlaps_1d_rect_partial() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(ispace.overlaps(&rect1d(5, 15)));
                assert!(ispace.overlaps(&rect1d(-5, 5)));
            }

            #[test]
            fn test_ispace_overlaps_1d_rect_encompassing() {
                let ispace = ispace_with_rect_1d(1, 2, 8);
                assert!(ispace.overlaps(&rect1d(0, 10)));
            }

            #[test]
            fn test_ispace_overlaps_1d_rect_touching() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(ispace.overlaps(&rect1d(10, 20)));
                assert!(ispace.overlaps(&rect1d(-10, 0)));
            }

            #[test]
            fn test_ispace_overlaps_1d_rect_disjoint() {
                let ispace = ispace_with_rect_1d(1, 0, 10);
                assert!(!ispace.overlaps(&rect1d(15, 20)));
                assert!(!ispace.overlaps(&rect1d(-20, -15)));
            }

            #[test]
            fn test_ispace_overlaps_1d_multiple_rects_first() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(1, &[0, 4], 1, NodeID(0));
                ispace.set_rect(1, &[10, 14], 1, NodeID(0));
                // Overlaps first rect only
                assert!(ispace.overlaps(&rect1d(2, 6)));
            }

            #[test]
            fn test_ispace_overlaps_1d_multiple_rects_second() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(1, &[0, 4], 1, NodeID(0));
                ispace.set_rect(1, &[10, 14], 1, NodeID(0));
                // Overlaps second rect only
                assert!(ispace.overlaps(&rect1d(12, 20)));
            }

            #[test]
            fn test_ispace_overlaps_1d_multiple_rects_gap() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(1, &[0, 4], 1, NodeID(0));
                ispace.set_rect(1, &[10, 14], 1, NodeID(0));
                // In the gap (within bounds but doesn't overlap any rect)
                assert!(!ispace.overlaps(&rect1d(6, 8)));
            }

            #[test]
            fn test_ispace_overlaps_1d_point_hit() {
                let ispace = ispace_with_point_1d(1, 5);
                assert!(ispace.overlaps(&rect1d(0, 10)));
                assert!(ispace.overlaps(&rect1d(5, 5)));
            }

            #[test]
            fn test_ispace_overlaps_1d_point_miss() {
                let ispace = ispace_with_point_1d(1, 5);
                assert!(!ispace.overlaps(&rect1d(0, 4)));
                assert!(!ispace.overlaps(&rect1d(6, 10)));
            }

            #[test]
            fn test_ispace_overlaps_2d_rect_inside() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(ispace.overlaps(&rect2d(2, 2, 8, 8)));
            }

            #[test]
            fn test_ispace_overlaps_2d_rect_partial() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(ispace.overlaps(&rect2d(5, 5, 15, 15)));
            }

            #[test]
            fn test_ispace_overlaps_2d_rect_disjoint() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(!ispace.overlaps(&rect2d(15, 15, 20, 20)));
            }

            #[test]
            fn test_ispace_overlaps_2d_rect_touching_corner() {
                let ispace = ispace_with_rect_2d(1, 0, 0, 10, 10);
                assert!(ispace.overlaps(&rect2d(10, 10, 20, 20)));
            }

            #[test]
            fn test_ispace_overlaps_2d_multiple_rects() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(2, &[0, 0, 4, 4], 2, NodeID(0));
                ispace.set_rect(2, &[10, 10, 14, 14], 2, NodeID(0));
                // Overlaps first
                assert!(ispace.overlaps(&rect2d(2, 2, 6, 6)));
                // Overlaps second
                assert!(ispace.overlaps(&rect2d(12, 12, 16, 16)));
                // In gap
                assert!(!ispace.overlaps(&rect2d(6, 6, 8, 8)));
            }

            #[test]
            fn test_ispace_overlaps_2d_point_hit() {
                let ispace = ispace_with_point_2d(1, 5, 5);
                assert!(ispace.overlaps(&rect2d(0, 0, 10, 10)));
                assert!(ispace.overlaps(&rect2d(5, 5, 5, 5)));
            }

            #[test]
            fn test_ispace_overlaps_2d_point_miss() {
                let ispace = ispace_with_point_2d(1, 5, 5);
                assert!(!ispace.overlaps(&rect2d(0, 0, 4, 4)));
                assert!(!ispace.overlaps(&rect2d(6, 6, 10, 10)));
            }

            #[test]
            fn test_ispace_overlaps_3d_rect_inside() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(ispace.overlaps(&rect3d(2, 2, 2, 8, 8, 8)));
            }

            #[test]
            fn test_ispace_overlaps_3d_rect_partial() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(ispace.overlaps(&rect3d(5, 5, 5, 15, 15, 15)));
            }

            #[test]
            fn test_ispace_overlaps_3d_rect_disjoint() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(!ispace.overlaps(&rect3d(15, 15, 15, 20, 20, 20)));
            }

            #[test]
            fn test_ispace_overlaps_3d_rect_touching_corner() {
                let ispace = ispace_with_rect_3d(1, 0, 0, 0, 10, 10, 10);
                assert!(ispace.overlaps(&rect3d(10, 10, 10, 20, 20, 20)));
            }

            #[test]
            fn test_ispace_overlaps_3d_multiple_rects() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(3, &[0, 0, 0, 4, 4, 4], 3, NodeID(0));
                ispace.set_rect(3, &[10, 10, 10, 14, 14, 14], 3, NodeID(0));
                // Overlaps first
                assert!(ispace.overlaps(&rect3d(2, 2, 2, 6, 6, 6)));
                // Overlaps second
                assert!(ispace.overlaps(&rect3d(12, 12, 12, 16, 16, 16)));
                // In gap
                assert!(!ispace.overlaps(&rect3d(6, 6, 6, 8, 8, 8)));
            }

            #[test]
            fn test_ispace_overlaps_3d_point_hit() {
                let ispace = ispace_with_point_3d(1, 5, 5, 5);
                assert!(ispace.overlaps(&rect3d(0, 0, 0, 10, 10, 10)));
                assert!(ispace.overlaps(&rect3d(5, 5, 5, 5, 5, 5)));
            }

            #[test]
            fn test_ispace_overlaps_3d_point_miss() {
                let ispace = ispace_with_point_3d(1, 5, 5, 5);
                assert!(!ispace.overlaps(&rect3d(0, 0, 0, 4, 4, 4)));
                assert!(!ispace.overlaps(&rect3d(6, 6, 6, 10, 10, 10)));
            }

            #[test]
            fn test_ispace_overlaps_3d_mixed() {
                let mut ispace = new_ispace(1);
                ispace.set_rect(3, &[0, 0, 0, 4, 4, 4], 3, NodeID(0));
                ispace.set_point(3, &[10, 10, 10], NodeID(0));
                // Overlaps rect
                assert!(ispace.overlaps(&rect3d(2, 2, 2, 6, 6, 6)));
                // Overlaps point
                assert!(ispace.overlaps(&rect3d(8, 8, 8, 12, 12, 12)));
                // Misses both
                assert!(!ispace.overlaps(&rect3d(6, 6, 6, 8, 8, 8)));
            }
        }
    }
}
