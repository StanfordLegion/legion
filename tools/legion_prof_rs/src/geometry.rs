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
                panic!("Unknown bounds");
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
                panic!("Unknown bounds");
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
                    panic!("Bad bounds entry");
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
                panic!("Bad index space bounds");
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
