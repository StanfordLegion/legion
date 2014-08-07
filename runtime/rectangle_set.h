/* Copyright 2014 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __LEGION_RECTANGLE_SET_H__
#define __LEGION_RECTANGLE_SET_H__

#include <set>
#include <list>
#include <vector>

#include <cstdio>
#include <cassert>

namespace LegionRuntime {
  namespace HighLevel {

    template<typename T>
    class Segment {
    public:
      enum Direction {
        LEFT_DIR = 0,
        RIGHT_DIR = 1,
        NONE_DIR = 2,
      };
    public:
      Segment(void);
      Segment(T a1, T a2, T b, Direction dir);
      Segment(const Segment &rhs);
      ~Segment(void);
    public:
      Segment& operator=(const Segment &rhs);
    public:
      inline bool intersects(const Segment<T> &other) const;
      inline bool touches(const Segment<T> &other) const;
      inline bool divides(const Segment<T> &other) const;
    public:
      inline void clear_adjacent(void);
      inline void clear_adjacent(T value);
      inline void remove_adjacent(Segment<T> *old);
      inline void add_adjacent(Segment<T> *seg);
      inline void set_adjacent(Segment<T> *one, Segment<T> *two);
      inline void replace_adjacent(Segment<T> *old_seg,
                                   Segment<T> *new_seg);
      inline bool has_adjacent(Segment<T> *seg) const;
      inline Segment<T>* find_adjoining(Segment<T> *par_seg, T value) const;
      inline Segment<T>* find_one_adjacent(T value) const;
      inline void move_adjacent(Segment<T> *target, T value);
      inline void sanity_check(void) const;
    public:
      inline void move_degenerate(Segment<T> *target);
      inline void move_degenerate(Segment<T> *target, T lower, T upper);
      inline void filter_degenerate(std::set<Segment<T>*> &segments);
      inline void filter_degenerate(T lower, T upper,
                                    std::set<Segment<T>*> &segments);
    public:
      inline bool points_left(void) const { return (dir == LEFT_DIR); }
      inline bool points_right(void) const { return (dir == RIGHT_DIR); }
      inline bool points_none(void) const { return (dir == NONE_DIR); }
    public:
      inline T distance_low(const Segment<T> &other) const;
      inline T distance_high(const Segment<T> &other) const;
    public:
      inline void add_reference(void);
      inline bool remove_reference(void);
    public:
      T a1, a2, b;
      Direction dir;
    protected:
      Segment<T> *adjacent_low[2];
      Segment<T> *adjacent_high[2];
      std::list<Segment<T>*> adjacent_deg;
      unsigned int references;
    };

    template<typename T>
    struct SplitSegment {
    public:
      SplitSegment(void)
        : segment(NULL), lower(NULL), higher(NULL) { }
      SplitSegment(Segment<T> *seg, Segment<T> *l, Segment<T> *h)
        : segment(seg), lower(l), higher(h) { }
    public:
      Segment<T> *segment;
      Segment<T> *lower, *higher;
    };

    template<typename T>
    struct RebuildRect {
    public:
      RebuildRect(T lx, T ly, T hx, T hy)
        : lower_x(lx), lower_y(ly),
          higher_x(hx), higher_y(hy) { }
    public:
      T lower_x, lower_y, higher_x, higher_y;
    };

    /**
     * \class RectangleSet
     * A class that represents a set of rectangles
     * and can be used for testing for domination of
     * another rectangle.
     */
    template<typename T, bool DISCRETE>
    class RectangleSet {
    public:
      RectangleSet(void);
      RectangleSet(const RectangleSet &rhs);
      ~RectangleSet(void);
    public:
      RectangleSet& operator=(const RectangleSet &rhs);
    public:
      inline void add_rectangle(T lower_x, T lower_y, T upper_x, T upper_y);
      inline bool covers(T lower_x, T lower_y, T upper_x, T upper_y) const;
    protected:
      static inline bool inside(const std::set<Segment<T>*> &segments, 
                                const std::set<Segment<T>*> &bounds,
                                const std::set<Segment<T>*> &other_bounds);
      static inline bool outside(const std::set<Segment<T>*> &segments,
                                 const std::set<Segment<T>*> &bounds);
      static inline void set_adjacent(const std::set<Segment<T>*> &xs,
                                      const std::set<Segment<T>*> &ys);
      static inline void compute_rebuild_rectangle(Segment<T> *current,
                                                   Segment<T> *next,
                                                   T current_line, T next_line,
                                        std::vector<RebuildRect<T> > &rebuilds,
                                                   T &min, T &max);
      static inline bool merge_adjacent(std::set<Segment<T>*> &segments,
                                        std::set<Segment<T>*> &other_segs,
                                        std::vector<RebuildRect<T> > &rebuilds);
      static inline bool handle_degenerate(Segment<T> *seg, T min, T max,
                                           std::set<Segment<T>*> &segments,
                                           std::set<Segment<T>*> &other_segs,
                                           bool add_next,
                                           std::vector<Segment<T>*> &next_segs);
      static Segment<T>* find_low(const Segment<T> &segment,
                                  const std::set<Segment<T>*> &bounds);
      static Segment<T>* find_high(const Segment<T> &segment,
                                   const std::set<Segment<T>*> &bounds);
      static bool has_divisor(const Segment<T> &segment,
                              const std::set<Segment<T>*> &bounds);
      static void boundary_edges(const std::set<Segment<T>*> &xs,
                                 const std::set<Segment<T>*> &xs_prime,
                                 const std::set<Segment<T>*> &ys_prime,
                                 std::set<Segment<T>*> &result);
      static void merge_segments(std::set<Segment<T>*> &segments);
      static void split_segment(Segment<T> *segment,
                                const std::set<Segment<T>*> &ys,
                                std::vector<SplitSegment<T> > &splits);
      static bool boundary(const SplitSegment<T> &segment);
      static bool has_overlap(Segment<T> *segment, 
                              const std::set<Segment<T>*> &bounds);
    protected:
      typename std::set<Segment<T>*> x_segments, y_segments;
    };

    /////////////////////////////////////////////////////////////
    // Segment 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    Segment<T>::Segment(void)
      : a1(0), a2(0), b(0), dir(LEFT_DIR), references(0)
    //--------------------------------------------------------------------------
    {
      clear_adjacent();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Segment<T>::Segment(T one, T two, T other, Direction d)
      : a1(one), a2(two), b(other), dir(d), references(0)
    //--------------------------------------------------------------------------
    {
      clear_adjacent();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Segment<T>::Segment(const Segment<T> &rhs)
      : a1(rhs.a1), a2(rhs.a2), b(rhs.b), dir(rhs.dir), references(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Segment<T>::~Segment(void)
    //--------------------------------------------------------------------------
    {
      if (adjacent_low[0] != NULL)
        adjacent_low[0]->remove_adjacent(this);
      if (adjacent_low[1] != NULL)
        adjacent_low[1]->remove_adjacent(this);
      if (adjacent_high[0] != NULL)
        adjacent_high[0]->remove_adjacent(this);
      if (adjacent_high[1] != NULL)
        adjacent_high[1]->remove_adjacent(this);
      for (typename std::list<Segment<T>*>::const_iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); it++)
      {
        (*it)->remove_adjacent(this);
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(references == 0);
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T>
    Segment<T>& Segment<T>::operator=(const Segment<T> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline bool Segment<T>::intersects(const Segment<T> &other) const
    //--------------------------------------------------------------------------
    {
      if (other.b <= a1)
        return false;
      if (other.b >= a2)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline bool Segment<T>::touches(const Segment<T> &other) const
    //--------------------------------------------------------------------------
    {
      if (other.b < a1)
        return false;
      if (other.b > a2)
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline bool Segment<T>::divides(const Segment<T> &other) const
    //--------------------------------------------------------------------------
    {
      // This is Segment Y
      // Other is Segment X
      return (touches(other) && other.intersects(*this));
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::clear_adjacent(void)
    //--------------------------------------------------------------------------
    {
      adjacent_low[0] = NULL;
      adjacent_low[1] = NULL;
      adjacent_high[0] = NULL;
      adjacent_high[1] = NULL;
      adjacent_deg.clear();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::clear_adjacent(T value)
    //--------------------------------------------------------------------------
    {
      if (value == a1)
      {
        if (adjacent_low[0] != NULL)
          adjacent_low[0]->remove_adjacent(this);
        adjacent_low[0] = NULL;
        if (adjacent_low[1] != NULL)
          adjacent_low[1]->remove_adjacent(this);
        adjacent_low[1] = NULL;
      }
      else if (value == a2)
      {
        if (adjacent_high[0] != NULL)
          adjacent_high[0]->remove_adjacent(this);
        adjacent_high[0] = NULL;
        if (adjacent_high[1] != NULL)
          adjacent_high[1]->remove_adjacent(this);
        adjacent_high[1] = NULL;
      }
      for (typename std::list<Segment<T>*>::iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); /*nothing*/)
      {
        if ((*it)->b == value)
          it = adjacent_deg.erase(it);
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::remove_adjacent(Segment<T> *old)
    //--------------------------------------------------------------------------
    {
      if (adjacent_low[0] == old)
        adjacent_low[0] = NULL;
      if (adjacent_low[1] == old)
        adjacent_low[1] = NULL;
      if (adjacent_high[0] == old)
        adjacent_high[0] = NULL;
      if (adjacent_high[1] == old)
        adjacent_high[1] = NULL;
      for (typename std::list<Segment<T>*>::iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); /*nothing*/)
      {
        if ((*it) == old)
          it = adjacent_deg.erase(it);
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::add_adjacent(Segment<T> *seg)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(seg != NULL);
      if (dir != NONE_DIR)
        assert((seg->a1 == b) || (seg->a2 == b));
      else
        assert((seg->a1 <= b) && (b <= seg->a2));
      assert((seg->b == a1) || (seg->b == a2) || seg->points_none());
#endif
      if (seg->points_none())
      {
        // Put it in the set of degenerate segments
        adjacent_deg.push_back(seg);
      }
      else if (seg->b == a1)
      {
#ifdef DEBUG_HIGH_LEVEL
        if (dir == LEFT_DIR)
        {
          if (seg->a1 == b)
            assert(seg->points_left() || seg->points_none());
          else
            assert(seg->points_right() || seg->points_none());
        }
        else if (dir == RIGHT_DIR)
        {
          if (seg->a1 == b)
            assert(seg->points_right() || seg->points_none());
          else
            assert(seg->points_left() || seg->points_none());
        }
        assert(adjacent_low[seg->dir] == NULL);
#endif
        adjacent_low[seg->dir] = seg;
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        if (dir == LEFT_DIR)
        {
          if (seg->a1 == b)
            assert(seg->points_right() || seg->points_none());
          else
            assert(seg->points_left() || seg->points_none());
        }
        else if (dir == RIGHT_DIR)
        {
          if (seg->a1 == b)
            assert(seg->points_left() || seg->points_none());
          else
            assert(seg->points_right() || seg->points_none());
        }
        assert(adjacent_high[seg->dir] == NULL);
#endif
        adjacent_high[seg->dir] = seg;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::set_adjacent(Segment<T> *one, Segment<T> *two)
    //--------------------------------------------------------------------------
    {
      clear_adjacent();
      add_adjacent(one);
      add_adjacent(two);
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    } 

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::replace_adjacent(Segment<T> *old_seg,
                                             Segment<T> *new_seg)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(old_seg != NULL);
      assert(new_seg != NULL);
#endif
      remove_adjacent(old_seg);
      add_adjacent(new_seg);
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline bool Segment<T>::has_adjacent(Segment<T> *seg) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
      if (adjacent_low[0] == seg)
        return true;
      if (adjacent_low[1] == seg)
        return true;
      if (adjacent_high[0] == seg)
        return true;
      if (adjacent_high[1] == seg)
        return true;
      for (typename std::list<Segment<T>*>::const_iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); it++)
      {
        if ((*it) == seg)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline Segment<T>* Segment<T>::find_adjoining(Segment<T> *par, 
                                                  T value) const
    //--------------------------------------------------------------------------
    {
      if (value == a1)
      {
        if ((adjacent_low[0] != NULL) &&
            (adjacent_low[0]->has_adjacent(par)))
          return adjacent_low[0];
        if ((adjacent_low[1] != NULL) &&
            (adjacent_low[1]->has_adjacent(par)))
          return adjacent_low[1];
      }
      else if (value == a2)
      {
        if ((adjacent_high[0] != NULL) &&
            (adjacent_high[0]->has_adjacent(par)))
          return adjacent_high[0];
        if ((adjacent_high[1] != NULL) &&
            (adjacent_high[1]->has_adjacent(par)))
          return adjacent_high[1];
      }
      for (typename std::list<Segment<T>*>::const_iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); it++)
      {
        if (((*it)->b == value) && (*it)->has_adjacent(par))
          return (*it);
      }
      return NULL;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline Segment<T>* Segment<T>::find_one_adjacent(T value) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert((value == a1) || (value == a2));
#endif
      if (value == a1)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(((adjacent_low[0] != NULL) && (adjacent_low[1] == NULL)) ||
               ((adjacent_low[0] == NULL) && (adjacent_low[1] != NULL)));
#endif
        if (adjacent_low[0] != NULL)
          return adjacent_low[0];
        else
          return adjacent_low[1];
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(((adjacent_high[0] != NULL) && (adjacent_high[1] == NULL)) ||
               ((adjacent_high[0] == NULL) && (adjacent_high[1] != NULL)));
#endif
        if (adjacent_high[0] != NULL)
          return adjacent_high[0];
        else
          return adjacent_high[1];
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::move_adjacent(Segment<T> *target, T value)
    //--------------------------------------------------------------------------
    {
      target->clear_adjacent(value);
      if (value == a1)
      {
        if (adjacent_low[0] != NULL)
        {
          target->add_adjacent(adjacent_low[0]);
          adjacent_low[0]->replace_adjacent(this, target);
          adjacent_low[0] = NULL;
        }
        if (adjacent_low[1] != NULL)
        {
          target->add_adjacent(adjacent_low[1]);
          adjacent_low[1]->replace_adjacent(this, target);
          adjacent_low[1] = NULL;
        }
      }
      else if (value == a2)
      {
        if (adjacent_high[0] != NULL)
        {
          target->add_adjacent(adjacent_high[0]);
          adjacent_high[0]->replace_adjacent(this, target);
          adjacent_high[0] = NULL;
        }
        if (adjacent_high[1] != NULL)
        {
          target->add_adjacent(adjacent_high[1]);
          adjacent_high[1]->replace_adjacent(this, target);
          adjacent_high[1] = NULL;
        }
      }
      for (typename std::list<Segment<T>*>::iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); /*nothing*/)
      {
        if ((*it)->b == value)
        {
          target->add_adjacent(*it);
          (*it)->replace_adjacent(this, target);
          it = adjacent_deg.erase(it);
        }
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::sanity_check(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(a1 <= a2);
      if (adjacent_low[0] != NULL)
      {
        if (dir != NONE_DIR)
          assert((adjacent_low[0]->a1 == b) || (adjacent_low[0]->a2 == b));
        else
          assert((adjacent_low[0]->a1 <= b) && (adjacent_low[0]->a2 >= b));
        if (dir != NONE_DIR)
          assert((adjacent_low[0]->b == a1) || (adjacent_low[0]->b == a2));
        else
          assert((a1 <= adjacent_low[0]->b) && (adjacent_low[0]->b <= a2));
      }
      if (adjacent_low[1] != NULL)
      {
        if (dir != NONE_DIR)
          assert((adjacent_low[1]->a1 == b) || (adjacent_low[1]->a2 == b));
        else
          assert((adjacent_low[1]->a1 <= b) && (adjacent_low[1]->a2 >= b));
        if (dir != NONE_DIR)
          assert((adjacent_low[1]->b == a1) || (adjacent_low[1]->b == a2));
        else
          assert((a1 <= adjacent_low[1]->b) && (adjacent_low[1]->b <= a2));
      }
      if (adjacent_high[0] != NULL)
      {
        if (dir != NONE_DIR)
          assert((adjacent_high[0]->a1 == b) || (adjacent_high[0]->a2 == b));
        else
          assert((adjacent_high[0]->a1 <= b) && (adjacent_high[0]->a2 >= b));
        if (dir != NONE_DIR)
          assert((adjacent_high[0]->b == a1) || (adjacent_high[0]->b == a2));
        else
          assert((a1 <= adjacent_high[0]->b) && (adjacent_high[0]->b <= a2));
      }
      if (adjacent_high[1] != NULL)
      {
        if (dir != NONE_DIR)
          assert((adjacent_high[1]->a1 == b) || (adjacent_high[1]->a2 == b));
        else
          assert((adjacent_high[1]->a1 <= b) && (adjacent_high[1]->a2 >= b));
        if (dir != NONE_DIR)
          assert((adjacent_high[1]->b == a1) || (adjacent_high[1]->b == a2));
        else
          assert((a1 <= adjacent_high[1]->b) && (adjacent_high[1]->b <= a2));
      }
      for (typename std::list<Segment<T>*>::const_iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); it++)
      {
        assert(((*it)->a1 <= b) && ((*it)->a2 >= b));
        assert((a1 <= (*it)->b) && ((*it)->b <= a2));
      }
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::move_degenerate(Segment<T> *target)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<Segment<T>*>::const_iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); it++)
      {
        target->add_adjacent(*it);
        (*it)->replace_adjacent(this, target);
      }
      adjacent_deg.clear();
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::move_degenerate(Segment<T> *target,
                                            T lower, T upper)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<Segment<T>*>::iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); /*nothing*/)
      {
        if ((lower <= (*it)->b) && ((*it)->b <= upper))
        {
          target->add_adjacent(*it);
          (*it)->replace_adjacent(this, target);
          it = adjacent_deg.erase(it);
        }
        else
          it++;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::filter_degenerate(std::set<Segment<T>*> &segments)
    //--------------------------------------------------------------------------
    {
      std::vector<Segment<T>*> to_erase(adjacent_deg.begin(),
                                        adjacent_deg.end());
      adjacent_deg.clear();
      for (typename std::vector<Segment<T>*>::const_iterator it = 
            to_erase.begin(); it != to_erase.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(segments.find(*it) != segments.end());
#endif
        segments.erase(*it);
        if ((*it)->remove_reference())
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::filter_degenerate(T lower, T upper,
                                              std::set<Segment<T>*> &segments)
    //--------------------------------------------------------------------------
    {
      // Need to do this in a separate pass to avoid 
      // callbacks from the deletion which invalidate the iterator
      std::vector<Segment<T>*> to_erase;
      for (typename std::list<Segment<T>*>::iterator it = 
            adjacent_deg.begin(); it != adjacent_deg.end(); /*nothing*/)
      {
        if ((lower <= (*it)->b) && ((*it)->b <= upper))
        {
          to_erase.push_back(*it);
          it = adjacent_deg.erase(it);
        }
        else
          it++;
      }
      for (typename std::vector<Segment<T>*>::const_iterator it =
            to_erase.begin(); it != to_erase.end(); it++)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(segments.find(*it) != segments.end());
#endif
        segments.erase(*it);
        if ((*it)->remove_reference())
          delete (*it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Segment<T>::distance_low(const Segment<T> &rhs) const
    //--------------------------------------------------------------------------
    {
      return (a1 - rhs.b);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline T Segment<T>::distance_high(const Segment<T> &rhs) const
    //--------------------------------------------------------------------------
    {
      return (a2 - rhs.b);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline void Segment<T>::add_reference(void)
    //--------------------------------------------------------------------------
    {
      references++;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline bool Segment<T>::remove_reference(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(references > 0);
#endif
      references--;
      return (references == 0);
    }

    /////////////////////////////////////////////////////////////
    // Rectangle Set 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    RectangleSet<T,DISCRETE>::RectangleSet(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    RectangleSet<T,DISCRETE>::RectangleSet(const RectangleSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    RectangleSet<T,DISCRETE>::~RectangleSet(void)
    //--------------------------------------------------------------------------
    {
      for (typename std::set<Segment<T>*>::iterator it = 
            x_segments.begin(); it != x_segments.end(); it++)
      {
        if ((*it)->remove_reference())
          delete (*it);
#ifdef DEBUG_HIGH_LEVEL
        else
          assert(false); // Memory leak
#endif
      }
      x_segments.clear();
      for (typename std::set<Segment<T>*>::const_iterator it = 
            y_segments.begin(); it != y_segments.end(); it++)
      {
        if ((*it)->remove_reference())
          delete (*it);
#ifdef DEBUG_HIGH_LEVEL
        else
          assert(false); // Memory leak
#endif
      }
      y_segments.clear();
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    RectangleSet<T,DISCRETE>& 
                    RectangleSet<T,DISCRETE>::operator=(const RectangleSet &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE> 
    inline void RectangleSet<T,DISCRETE>::add_rectangle(T lower_x, T lower_y, 
                                                        T upper_x, T upper_y)
    //--------------------------------------------------------------------------
    { 
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_x <= upper_x);
      assert(lower_y <= upper_y);
#endif
      if (!DISCRETE && ((lower_x == upper_x) || (lower_y == upper_y)))
        return;
      if (!DISCRETE || ((lower_x < upper_x) && (lower_y < upper_y)))
      {
        std::set<Segment<T>*> north_south, east_west, new_xs, new_ys;
        Segment<T> *north = new Segment<T>(lower_x, upper_x, 
                                           upper_y, Segment<T>::LEFT_DIR);
        Segment<T> *south = new Segment<T>(lower_x, upper_x,
                                           lower_y, Segment<T>::RIGHT_DIR);
        Segment<T> *east = new Segment<T>(lower_y, upper_y,
                                          upper_x, Segment<T>::LEFT_DIR);
        Segment<T> *west = new Segment<T>(lower_y, upper_y,
                                          lower_x, Segment<T>::RIGHT_DIR);
        north_south.insert(north);
        north->add_reference();
        north->set_adjacent(east, west);
        north_south.insert(south);
        south->add_reference();
        south->set_adjacent(east, west);
        east_west.insert(east);
        east->add_reference();
        east->set_adjacent(north, south);
        east_west.insert(west);
        west->add_reference();
        west->set_adjacent(north, south);

        boundary_edges(north_south, x_segments, y_segments, new_xs);
        boundary_edges(x_segments, north_south, east_west, new_xs);
        boundary_edges(east_west, y_segments, x_segments, new_ys);
        boundary_edges(y_segments, east_west, north_south, new_ys);
        // Merge the new sets of segments
        merge_segments(new_xs);
        merge_segments(new_ys);
        // Clean out the x_segments and y_segments 
        // and update them to point to the new segments
        for (typename std::set<Segment<T>*>::const_iterator it = 
              x_segments.begin(); it != x_segments.end(); it++)
        {
          if ((*it)->remove_reference())
            delete (*it);
        }
        x_segments = new_xs;
        for (typename std::set<Segment<T>*>::const_iterator it = 
              y_segments.begin(); it != y_segments.end(); it++)
        {
          if ((*it)->remove_reference())
            delete (*it);
        }
        y_segments = new_ys;
        if (north->remove_reference())
          delete north;
        if (south->remove_reference())
          delete south;
        if (east->remove_reference())
          delete east;
        if (west->remove_reference())
          delete west;
      }
      else if (DISCRETE)
      {
        // If we are here, we have a degenerate rectangle
        if (lower_x == upper_x)
        {
          Segment<T> *new_degenerate = 
            new Segment<T>(lower_y, upper_y, lower_x, Segment<T>::NONE_DIR);
          std::set<Segment<T>*> deg_set, new_xs, new_ys, empty_set;
          new_degenerate->add_reference();
          deg_set.insert(new_degenerate);
          boundary_edges(deg_set, y_segments, x_segments, new_ys);
          boundary_edges(y_segments, deg_set, empty_set, new_ys);
          boundary_edges(x_segments, empty_set, deg_set, new_xs);
          merge_segments(new_xs);
          merge_segments(new_ys);
          for (typename std::set<Segment<T>*>::const_iterator it = 
                x_segments.begin(); it != x_segments.end(); it++)
          {
            if ((*it)->remove_reference())
              delete (*it);
          }
          x_segments = new_xs;
          for (typename std::set<Segment<T>*>::const_iterator it = 
                y_segments.begin(); it != y_segments.end(); it++)
          {
            if ((*it)->remove_reference())
              delete (*it);
          }
          y_segments = new_ys;
          if (new_degenerate->remove_reference())
            delete new_degenerate;
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(lower_y == upper_y);
#endif
          Segment<T> *new_degenerate = 
            new Segment<T>(lower_x, upper_x, lower_y, Segment<T>::NONE_DIR);
          std::set<Segment<T>*> deg_set, new_xs, new_ys, empty_set;
          new_degenerate->add_reference();
          deg_set.insert(new_degenerate);
          boundary_edges(deg_set, x_segments, y_segments, new_xs);
          boundary_edges(x_segments, deg_set, empty_set, new_xs);
          boundary_edges(y_segments, empty_set, deg_set, new_ys);
          merge_segments(new_xs);
          merge_segments(new_ys);
          for (typename std::set<Segment<T>*>::const_iterator it = 
                x_segments.begin(); it != x_segments.end(); it++)
          {
            if ((*it)->remove_reference())
              delete (*it);
          }
          x_segments = new_xs;
          for (typename std::set<Segment<T>*>::const_iterator it = 
                y_segments.begin(); it != y_segments.end(); it++)
          {
            if ((*it)->remove_reference())
              delete (*it);
          }
          y_segments = new_ys;
          if (new_degenerate->remove_reference())
            delete new_degenerate;
        }
      }
      // Rebuild the adjacent sets
      set_adjacent(x_segments, y_segments);
      // Finally if this is a discrete case, we have to filter the
      // segments which are adjacent to each other in space (off by one)
      if (DISCRETE && !x_segments.empty() && !y_segments.empty())
      {
        std::vector<RebuildRect<T> > rebuild_x, rebuild_y;
        bool changed = true;
        while (changed)
        {
          bool change_x = merge_adjacent(x_segments, y_segments, rebuild_x);
          bool change_y = merge_adjacent(y_segments, x_segments, rebuild_y);
          if (!rebuild_x.empty())
          {
            for (typename std::vector<RebuildRect<T> >::const_iterator it = 
                  rebuild_x.begin(); it != rebuild_x.end(); it++)
            {
              // Need a transpose here to get the dimensions correct
              add_rectangle(it->lower_y, it->lower_x, 
                            it->higher_y, it->higher_x);
            }
            rebuild_x.clear();
          }
          if (!rebuild_y.empty())
          {
            for (typename std::vector<RebuildRect<T> >::const_iterator it = 
                  rebuild_y.begin(); it != rebuild_y.end(); it++)
            {
              // Need a transpose here to get the dimensions correct
              add_rectangle(it->lower_x, it->lower_y,
                            it->higher_x, it->higher_y);
            }
            rebuild_y.clear();
          }
          changed = change_x || change_y;
        }
      }
    }
    
    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    inline bool RectangleSet<T,DISCRETE>::covers(T lower_x, T lower_y, 
                                                 T upper_x, T upper_y) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(lower_x <= upper_x);
      assert(lower_y <= upper_y);
#endif
      if (!DISCRETE && ((lower_x == upper_x) || (lower_y == upper_y)))
        return true;
      if (!DISCRETE || ((lower_x < upper_x) && (lower_y < upper_y)))
      {
        Segment<T> north(lower_x, upper_x, upper_y, Segment<T>::LEFT_DIR);
        Segment<T> south(lower_x, upper_x, lower_y, Segment<T>::RIGHT_DIR);
        Segment<T> east(lower_y, upper_y, upper_x, Segment<T>::LEFT_DIR);
        Segment<T> west(lower_y, upper_y, lower_x, Segment<T>::RIGHT_DIR);
        north.set_adjacent(&east, &west);
        south.set_adjacent(&east, &west);
        east.set_adjacent(&north, &south);
        west.set_adjacent(&north, &south);
        north.add_reference();
        south.add_reference();
        east.add_reference();
        west.add_reference();
        std::set<Segment<T>*> north_south, east_west;
        north_south.insert(&north);
        north_south.insert(&south);
        east_west.insert(&east);
        east_west.insert(&west);
        bool result;
        if (!inside(north_south, y_segments, x_segments))
          result = false;
        else if (!inside(east_west, x_segments, y_segments))
          result = false;
        else if (!outside(x_segments, east_west))
          result = false;
        else if (!outside(y_segments, north_south))
          result = false;
        else
          result = true;
        north.remove_reference();
        south.remove_reference();
        east.remove_reference();
        west.remove_reference();
        return result;
      }
      else if (DISCRETE)
      {
        if (lower_x == upper_x)
        {
          // Handle the super special case of testing a single point
          if (lower_y == upper_y)
          {
            Segment<T> degenerate_h(lower_x, upper_x, lower_y,
                                    Segment<T>::NONE_DIR);
            Segment<T> degenerate_v(lower_y, upper_y, lower_x,
                                    Segment<T>::NONE_DIR);
            degenerate_h.add_reference();
            degenerate_v.add_reference();
            std::set<Segment<T>*> h_set, v_set;
            h_set.insert(&degenerate_h);
            v_set.insert(&degenerate_v);
            bool result;
            if (!inside(v_set, x_segments, y_segments) && 
                !inside(h_set, y_segments, x_segments))
              result = false;
            else if (!outside(x_segments, v_set) && 
                     !outside(y_segments, h_set))
              result = false;
            else
              result = true;
            degenerate_h.remove_reference();
            degenerate_v.remove_reference();
            return result;
          }
          else
          {
            Segment<T> degenerate(lower_y, upper_y, lower_x, 
                                  Segment<T>::NONE_DIR);
            std::set<Segment<T>*> deg_set;
            degenerate.add_reference();
            deg_set.insert(&degenerate);
            bool result;
            if (!inside(deg_set, x_segments, y_segments))
              result = false;
            else if (!outside(x_segments, deg_set))
              result = false;
            else
              result = true;
            degenerate.remove_reference();
            return result;
          }
        }
        else
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(lower_y == upper_y);
#endif
          Segment<T> degenerate(lower_x, upper_x, lower_y,
                                Segment<T>::NONE_DIR);
          std::set<Segment<T>*> deg_set;
          degenerate.add_reference();
          deg_set.insert(&degenerate);
          bool result;
          if (!inside(deg_set, y_segments, x_segments))
            result = false;
          else if (!outside(y_segments, deg_set))
            result = false;
          else
            result = true;
          degenerate.remove_reference();
          return result;
        }
      }
      else
      {
        // should never get here
        assert(false);
        return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline bool RectangleSet<T,DISCRETE>::inside(
     const std::set<Segment<T>*> &segments, const std::set<Segment<T>*> &bounds,
     const std::set<Segment<T>*> &other_bounds)
    //--------------------------------------------------------------------------
    {
      std::vector<SplitSegment<T> > split_segments;
      for (typename std::set<Segment<T>*>::const_iterator it = 
            segments.begin(); it != segments.end(); it++)
      {
        split_segment((*it), bounds, split_segments);
      }
      bool result = true;
      for (typename std::vector<SplitSegment<T> >::const_iterator it =
            split_segments.begin(); it != split_segments.end(); it++)
      {
        if (boundary(*it))
        {
          Segment<T> *low = it->lower;
          Segment<T> *high = it->higher;
          if ((low == NULL) || (high == NULL))
          {
            // Special case for handling degenerate rectangles
            if (it->segment->points_none() &&
                has_overlap(it->segment, other_bounds))
              continue;
            result = false;
            break;
          }
          Segment<T> *adjoining = low->find_adjoining(high, it->segment->b);
          if ((adjoining != NULL) &&
              (adjoining->a1 <= it->segment->a1) &&
              (adjoining->a2 >= it->segment->a2))
          {
            if ((it->segment->dir != adjoining->dir) &&
                (!it->segment->points_none()))
            {
              result = false;
              break;
            }
          }
          else if (!low->points_right())
          {
            // Last check for overlapping for degenerates
            if (it->segment->points_none() && 
                has_overlap(it->segment, other_bounds))
              continue;
            result = false;
            break;
          }
        }
      }
      // Cleanup our mess
      for (typename std::vector<SplitSegment<T> >::const_iterator it = 
            split_segments.begin(); it != split_segments.end(); it++)
      {
        if (it->segment->remove_reference())
          delete it->segment;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline bool RectangleSet<T,DISCRETE>::outside(
     const std::set<Segment<T>*> &segments, const std::set<Segment<T>*> &bounds)
    //--------------------------------------------------------------------------
    {
      bool result = true;
      for (typename std::set<Segment<T>*>::const_iterator seg_it = 
            segments.begin(); result && (seg_it != segments.end()); seg_it++)
      {
        std::vector<SplitSegment<T> > split_segments;
        split_segment(*seg_it, bounds, split_segments);
        for (typename std::vector<SplitSegment<T> >::const_iterator it = 
              split_segments.begin(); it != split_segments.end(); it++)
        {
          Segment<T> *low = it->lower;
          if ((low == NULL) || low->points_none())
            continue;
          Segment<T> *high = it->higher;
          if ((high == NULL) || high->points_none())
            continue;
          Segment<T> *adjoining = low->find_adjoining(high, it->segment->b);
          if ((adjoining != NULL) &&
              (adjoining->a1 <= it->segment->a1) && 
              (adjoining->a2 >= it->segment->a2))
          {
            continue;
          }
          result = false;
          break;
        }
        // clean up our segments
        for (typename std::vector<SplitSegment<T> >::const_iterator it = 
              split_segments.begin(); it != split_segments.end(); it++)
        {
          if (it->segment->remove_reference())
            delete it->segment;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline void RectangleSet<T,DISCRETE>::set_adjacent(
               const std::set<Segment<T>*> &xs, const std::set<Segment<T>*> &ys)
    //--------------------------------------------------------------------------
    {
      for (typename std::set<Segment<T>*>::const_iterator it = xs.begin();
            it != xs.end(); it++)
      {
        (*it)->clear_adjacent();
      }
      for (typename std::set<Segment<T>*>::const_iterator it = ys.begin();
            it != ys.end(); it++)
      {
        (*it)->clear_adjacent();
      }
      for (typename std::set<Segment<T>*>::const_iterator it = xs.begin();
            it != xs.end(); it++)
      {
        Segment<T> *x = (*it);
        if (x->points_none())
        {
          for (typename std::set<Segment<T>*>::const_iterator it2 = ys.begin();
                it2 != ys.end(); it2++)
          {
            Segment<T> *y = (*it2);
            if (y->points_none())
            {
              // Both point to none
              if ((x->a1 <= y->b) && (y->b <= x->a2) &&
                  (y->a1 <= x->b) && (x->b <= y->a2))
              {
                x->add_adjacent(y);
                y->add_adjacent(x);
              }
            }
            else
            {
              // x points to none
              if (((x->a1 == y->b) || (x->a2 == y->b)) &&
                  (y->a1 <= x->b) && (x->b <= y->a2))
              {
                x->add_adjacent(y);
                y->add_adjacent(x);
              }
            }
          }
        }
        else
        {
          for (typename std::set<Segment<T>*>::const_iterator it2 = ys.begin();
                it2 != ys.end(); it2++)
          {
            Segment<T> *y = (*it2);
            if (y->points_none())
            {
              // y points to none
              if (((y->a1 == x->b) || (y->a2 == x->b)) &&
                  (x->a1 <= y->b) && (y->b <= x->a2))
              {
                x->add_adjacent(y);
                y->add_adjacent(x);
              }
            }
            else
            {
              // neither points to none
              if (((x->a1 == y->b) || (x->a2 == y->b)) &&
                  ((y->a1 == x->b) || (y->a2 == x->b)))
              {
                x->add_adjacent(y);
                y->add_adjacent(x);
              }
            }
          }
        }
      }
    }
 
    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline void RectangleSet<T,DISCRETE>::compute_rebuild_rectangle(
                          Segment<T> *current, Segment<T> *next, 
                          T current_line, T next_line,
                          std::vector<RebuildRect<T> >&rebuilds, T &min, T &max)
    //--------------------------------------------------------------------------
    {
      min = (current->a1 < next->a1) ? 
             next->a1 : current->a1;
      max = (current->a2 > next->a2) ?
             next->a2 : current->a2;
#ifdef DEBUG_HIGH_LEVEL
      assert(min <= max);
#endif
      if (min < max)
        rebuilds.push_back(RebuildRect<T>(current_line, min, next_line, max));
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline bool RectangleSet<T,DISCRETE>::handle_degenerate(
                 Segment<T> *seg, T min, T max, std::set<Segment<T>*> &segments,
                 std::set<Segment<T>*> &other_segs, bool add_next, 
                 std::vector<Segment<T>*> &next_segments)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      seg->sanity_check();
#endif
      if ((min <= seg->a1) && (max >= seg->a2))
      {
        // Dominated so we can delete it
#ifdef DEBUG_HIGH_LEVEL
        assert(segments.find(seg) != segments.end());
#endif
        seg->filter_degenerate(other_segs);
        segments.erase(seg);
        if (seg->remove_reference())
          delete seg;
        return true;
      }
      else if ((seg->a1 < min) && (seg->a2 > max))
      {
        // Split into two pieces
        Segment<T> *new_seg =
          new Segment<T>(seg->a1, min, seg->b, Segment<T>::NONE_DIR);
        new_seg->add_reference();
        seg->move_adjacent(new_seg, seg->a1);
        seg->move_degenerate(new_seg, seg->a1, min);
        segments.insert(new_seg);
        if (add_next)
          next_segments.push_back(new_seg);
        seg->filter_degenerate(min, max, other_segs);
        seg->a1 = max;
        return false;
      }
      else if ((seg->a2 > max) && (seg->a1 < max))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(seg->a1 >= min);
#endif
        // Condense right
        seg->filter_degenerate(seg->a1, max, other_segs);
        seg->clear_adjacent(seg->a1);
        seg->a1 = max;
        return false;
      }
      else if ((seg->a1 < min) && (seg->a2 > min))
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(seg->a2 <= max);
#endif
        // Condense left
        seg->filter_degenerate(min, seg->a2, other_segs);
        seg->clear_adjacent(seg->a2);
        seg->a2 = min;
        return false;
      }
      else
      {
        // Otherwise they just touch on the ends
        return false;
      }
    }

    template<typename T>
    struct MergeComparator {
    public:
      bool operator()(const Segment<T> *left, const Segment<T> *right)
      {
        if (left->b < right->b)
          return true;
        else if (left->b > right->b)
          return false;
        else
        {
          return (left < right);
        }
      }
    };

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ inline bool RectangleSet<T,DISCRETE>::merge_adjacent(
             std::set<Segment<T>*> &segments, std::set<Segment<T>*> &other_segs,
             std::vector<RebuildRect<T> > &rebuild_rects)
    //--------------------------------------------------------------------------
    {
      std::set<Segment<T>*,MergeComparator<T> > sorted_segs(segments.begin(),
                                                            segments.end());
#ifdef DEBUG_HIGH_LEVEL
      assert(sorted_segs.size() == segments.size());
#endif
      bool result = false;
      std::vector<Segment<T>*> current_segments;
      typename std::set<Segment<T>*>::const_iterator it = sorted_segs.begin();
      current_segments.push_back(*it);
      T current_line = (*it)->b;
      it++;
      while ((it != sorted_segs.end()) && ((*it)->b == current_line))
      {
        current_segments.push_back(*it);
        it++;
      }
      // Now we have our first batch of segments on the same line
      while (it != sorted_segs.end())
      {
#ifdef DEBUG_HIGH_LEVEL
        assert((*it)->b > current_line);
#endif
        // Build the next line
        T next_line = (*it)->b;
        std::vector<Segment<T>*> next_segments;
        next_segments.push_back(*it);
        it++;
        while ((it != sorted_segs.end()) && ((*it)->b == next_line))
        {
          next_segments.push_back(*it);
          it++;
        }
        if ((current_line+1) == next_line)
        {
          std::vector<Segment<T>*> to_add;
          std::set<Segment<T>*> to_remove;
          // See if any of them overlap with each other
          for (typename std::vector<Segment<T>*>::const_iterator curr_it = 
                current_segments.begin(); curr_it != 
                current_segments.end(); curr_it++)
          {
            for (typename std::vector<Segment<T>*>::const_iterator next_it = 
                  next_segments.begin(); next_it != 
                  next_segments.end(); next_it++)
            {
              // First we have to check skip any segments
              // which are no longer valid
              if (to_remove.find(*next_it) != to_remove.end())
                continue;
              // Quick tests for non-overlap
              else if ((*curr_it)->a1 > (*next_it)->a2)
                continue;
              else if ((*curr_it)->a2 < (*next_it)->a1)
                continue;
              // Handle some degenerate cases
              else if ((*curr_it)->points_none() && 
                       (*next_it)->points_none())
              {
                T min, max;                  
                compute_rebuild_rectangle((*curr_it), (*next_it),
                                           current_line, next_line,
                                           rebuild_rects, min, max);
                if (handle_degenerate((*next_it), min, max, 
                                      segments, other_segs,
                                      true/*add next*/, to_add))
                  to_remove.insert(*next_it);
                if (handle_degenerate((*curr_it), min, max, 
                                      segments, other_segs,
                                      false/*add next*/, to_add))
                  break;
              }
              else if ((*curr_it)->points_none())
              {
                T min, max;                  
                compute_rebuild_rectangle((*curr_it), (*next_it),
                                           current_line, next_line,
                                           rebuild_rects, min, max);
                if (handle_degenerate((*curr_it), min, max, 
                                      segments, other_segs,
                                      false/*add next*/, to_add))
                  break;
              }
              else if ((*next_it)->points_none())
              {
                T min, max;                  
                compute_rebuild_rectangle((*curr_it), (*next_it),
                                           current_line, next_line,
                                           rebuild_rects, min, max);
                if (handle_degenerate((*next_it), min, max, 
                                      segments, other_segs,
                                      true/*add next*/, to_add))
                  to_remove.insert(*next_it);
              }
              // They also need to point away from each other
              else if ((*curr_it)->points_right() ||
                       (*next_it)->points_left())
                continue;
              // First case, equality
              else if (((*curr_it)->a1 == (*next_it)->a1) && 
                       ((*curr_it)->a2 == (*next_it)->a2))
              {
                result = true;
#ifdef DEBUG_HIGH_LEVEL
                assert((*curr_it)->dir != (*next_it)->dir);
#endif                
                // There are four possibilites here
                // 1. Share both edges
                // 2. Share only left
                // 3. Share only right
                // 4. Don't share either
                Segment<T> *adj_left = 
                  (*curr_it)->find_adjoining((*next_it), (*curr_it)->a1); 
                Segment<T> *adj_right = 
                  (*curr_it)->find_adjoining((*next_it), (*curr_it)->a2); 
                if ((adj_left != NULL) && (adj_right != NULL))
                {
                  // This is the easy case, everything
                  // gets deleted
#ifdef DEBUG_HIGH_LEVEL
                  assert(segments.find(*curr_it) != segments.end());
                  assert(segments.find(*next_it) != segments.end());
                  assert(other_segs.find(adj_left) != other_segs.end());
                  assert(other_segs.find(adj_right) != other_segs.end());
#endif
                  (*curr_it)->filter_degenerate(other_segs);
                  segments.erase(*curr_it);
                  if ((*curr_it)->remove_reference())
                    delete (*curr_it);
                  (*next_it)->filter_degenerate(other_segs);
                  segments.erase(*next_it);
                  if ((*next_it)->remove_reference())
                    delete (*next_it);
                  adj_left->filter_degenerate(other_segs);
                  other_segs.erase(adj_left);
                  if (adj_left->remove_reference())
                    delete adj_left;
                  adj_right->filter_degenerate(other_segs);
                  other_segs.erase(adj_right);
                  if (adj_right->remove_reference())
                    delete adj_right;
                  // Be sure to not include the next segment
                  to_remove.insert(*next_it);
                }
                else if (adj_left != NULL)
                {
                  Segment<T> *curr_right = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a2);
                  Segment<T> *next_right = 
                    (*next_it)->find_one_adjacent((*next_it)->a2);
#ifdef DEBUG_HIGH_LEVEL
                  assert(curr_right->dir == next_right->dir);
#endif
                  // Share the left edge, so merge right
                  // and get rid of left and current and next
                  curr_right->a2 = next_right->a2;
                  next_right->move_adjacent(curr_right, next_right->a2);
                  next_right->move_degenerate(curr_right);
#ifdef DEBUG_HIGH_LEVEL
                  assert(segments.find(*curr_it) != segments.end());
                  assert(segments.find(*next_it) != segments.end());
                  assert(other_segs.find(adj_left) != other_segs.end());
                  assert(other_segs.find(next_right) != other_segs.end());
                  curr_right->sanity_check();
#endif
                  (*curr_it)->filter_degenerate(other_segs);
                  segments.erase(*curr_it);
                  if ((*curr_it)->remove_reference())
                    delete (*curr_it);
                  (*next_it)->filter_degenerate(other_segs);
                  segments.erase(*next_it);
                  if ((*next_it)->remove_reference())
                    delete (*next_it);
                  adj_left->filter_degenerate(other_segs);
                  other_segs.erase(adj_left);
                  if (adj_left->remove_reference())
                    delete adj_left;
                  other_segs.erase(next_right);
                  if (next_right->remove_reference())
                    delete next_right;
                  // Be sure to not include in the next segment
                  to_remove.insert(*next_it);
                }
                else if (adj_right != NULL)
                {
                  Segment<T> *curr_left = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a1);
                  Segment<T> *next_left = 
                    (*next_it)->find_one_adjacent((*next_it)->a1);
#ifdef DEBUG_HIGH_LEVEL
                  assert(curr_left->dir == next_left->dir);
#endif
                  // Share the right edge, so merge left
                  // and get rid of right and current and next
                  curr_left->a2 = next_left->a2;
                  next_left->move_adjacent(curr_left, next_left->a2);
                  next_left->move_degenerate(curr_left);
#ifdef DEBUG_HIGH_LEVEL
                  assert(segments.find(*curr_it) != segments.end());
                  assert(segments.find(*next_it) != segments.end());
                  assert(other_segs.find(adj_right) != other_segs.end());
                  assert(other_segs.find(next_left) != other_segs.end());
                  curr_left->sanity_check();
#endif
                  (*curr_it)->filter_degenerate(other_segs);
                  segments.erase(*curr_it);
                  if ((*curr_it)->remove_reference())
                    delete (*curr_it);
                  (*next_it)->filter_degenerate(other_segs);
                  segments.erase(*next_it);
                  if ((*next_it)->remove_reference())
                    delete (*next_it);
                  adj_right->filter_degenerate(other_segs);
                  other_segs.erase(adj_right);
                  if (adj_right->remove_reference())
                    delete adj_right;
                  other_segs.erase(next_left);
                  if (next_left->remove_reference())
                    delete next_left;
                  // Be sure to not include in the next segment
                  to_remove.insert(*next_it);
                }
                else
                {
                  Segment<T> *curr_left = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a1);
                  Segment<T> *curr_right = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a2);
                  Segment<T> *next_left = 
                    (*next_it)->find_one_adjacent((*next_it)->a1);
                  Segment<T> *next_right = 
                    (*next_it)->find_one_adjacent((*next_it)->a2);
#ifdef DEBUG_HIGH_LEVEL
                  assert(curr_left->dir == next_left->dir);
                  assert(curr_right->dir == next_right->dir);
#endif
                  // Extend current left and current right
                  // Delete next left and next right
                  // Delete current and next
                  curr_left->a2 = next_left->a2;
                  next_left->move_adjacent(curr_left, next_left->a2);
                  next_left->move_degenerate(curr_left);
                  curr_right->a2 = next_right->a2;
                  next_right->move_adjacent(curr_right, next_right->a2);
                  next_right->move_degenerate(curr_right);
                  // Delete everything else
#ifdef DEBUG_HIGH_LEVEL
                  assert(segments.find(*curr_it) != segments.end());
                  assert(segments.find(*next_it) != segments.end());
                  assert(other_segs.find(next_left) != other_segs.end());
                  assert(other_segs.find(next_right) != other_segs.end());
                  curr_left->sanity_check();
                  curr_right->sanity_check();
#endif
                  (*curr_it)->filter_degenerate(other_segs);
                  segments.erase(*curr_it);
                  if ((*curr_it)->remove_reference())
                    delete (*curr_it);
                  (*next_it)->filter_degenerate(other_segs);
                  segments.erase(*next_it);
                  if ((*next_it)->remove_reference())
                    delete (*next_it);
                  other_segs.erase(next_left);
                  if (next_left->remove_reference())
                    delete next_left;
                  other_segs.erase(next_right);
                  if (next_right->remove_reference())
                    delete next_right;
                  // Be sure to not include the next segment
                  to_remove.insert(*next_it);
                }
                // We perfectly matched a segment in next so 
                // we can go onto the next current segment
                // Break out of the inner loop
                break;
              }
              // Next case domination by current
              else if (((*curr_it)->a1 <= (*next_it)->a1) &&
                       ((*curr_it)->a2 >= (*next_it)->a2))
              {
                result = true;
#ifdef DEBUG_HIGH_LEVEL
                assert((*curr_it)->dir != (*next_it)->dir);
#endif
                if ((*curr_it)->a1 == (*next_it)->a1)
                {
                  // Same on left
                  // This part is the same regardless of the left edge
                  Segment<T> *right_edge = 
                    (*next_it)->find_one_adjacent((*next_it)->a2);
                  // See if they are adjoining or not
                  Segment<T> *adj = 
                    (*curr_it)->find_adjoining((*next_it), (*curr_it)->a1);
                  (*curr_it)->filter_degenerate((*curr_it)->a1, 
                                                (*next_it)->a2, other_segs);
                  right_edge->a1 = current_line;
                  (*curr_it)->a1 = (*next_it)->a2;
                  right_edge->clear_adjacent(right_edge->a1);
                  right_edge->replace_adjacent((*next_it), (*curr_it));
                  if (adj != NULL)
                  {
                    // Only one adjacent edge
                    // Down scale current, extend next right
                    // and delete adj
                    (*curr_it)->clear_adjacent((*curr_it)->a1);
                    (*curr_it)->replace_adjacent(adj, right_edge);
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(adj) != other_segs.end());
#endif
                    adj->filter_degenerate(segments);
                    other_segs.erase(adj);
                    if (adj->remove_reference())
                      delete adj;
                  }
                  else
                  {
                    // Two separate edges that we need to merge
                    Segment<T> *curr_left = 
                      (*curr_it)->find_one_adjacent((*curr_it)->a1);
                    Segment<T> *next_left = 
                      (*next_it)->find_one_adjacent((*next_it)->a1);
#ifdef DEBUG_HIGH_LEVEL
                    assert(curr_left->dir == next_left->dir);
#endif
                    (*curr_it)->clear_adjacent((*curr_it)->a1);
                    (*curr_it)->replace_adjacent(curr_left, right_edge);
                    // Keep the current left edge
                    curr_left->a2 = next_left->a2;
                    next_left->move_adjacent(curr_left, next_left->a2);
                    next_left->move_degenerate(curr_left);
                    // Now we can delete next left
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(next_left) != other_segs.end());
#endif
                    other_segs.erase(next_left);
                    if (next_left->remove_reference())
                      delete next_left;
                  }
#ifdef DEBUG_HIGH_LEVEL
                  right_edge->sanity_check();
                  (*curr_it)->sanity_check();
#endif
                }
                else if ((*curr_it)->a2 == (*next_it)->a2)
                {
                  // Same on right
                  // This part is the same regardless of the right edge
                  Segment<T> *left_edge = 
                    (*next_it)->find_one_adjacent((*next_it)->a1);
                  // See if they are adjoining or not
                  Segment<T> *adj = 
                    (*curr_it)->find_adjoining((*next_it), (*curr_it)->a2);
                  (*curr_it)->filter_degenerate((*next_it)->a1, 
                                                (*curr_it)->a2, other_segs);
                  left_edge->a1 = current_line;
                  (*curr_it)->a2 = (*next_it)->a1;
                  left_edge->clear_adjacent(left_edge->a1);
                  left_edge->replace_adjacent((*next_it), (*curr_it));
                  if (adj != NULL)
                  {
                    (*curr_it)->clear_adjacent((*curr_it)->a2);
                    (*curr_it)->replace_adjacent(adj, left_edge);
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(adj) != other_segs.end());
#endif
                    adj->filter_degenerate(segments);
                    other_segs.erase(adj);
                    if (adj->remove_reference())
                      delete adj;
                  }
                  else
                  {
                    // Two separate edges that we need to merge
                    Segment<T> *curr_right = 
                      (*curr_it)->find_one_adjacent((*curr_it)->a2);
                    Segment<T> *next_right = 
                      (*next_it)->find_one_adjacent((*next_it)->a2);
#ifdef DEBUG_HIGH_LEVEL
                    assert(curr_right->dir == next_right->dir);
#endif
                    (*curr_it)->clear_adjacent((*curr_it)->a2);
                    (*curr_it)->replace_adjacent(curr_right, left_edge);
                    // Keep the current right edge
                    curr_right->a2 = next_right->a2;
                    next_right->move_adjacent(curr_right, next_right->a2);
                    next_right->move_degenerate(curr_right);
                    // Now we can delete next right
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(next_right) != other_segs.end());
#endif
                    other_segs.erase(next_right);
                    if (next_right->remove_reference())
                      delete next_right;
                  }
#ifdef DEBUG_HIGH_LEVEL
                  left_edge->sanity_check();
                  (*curr_it)->sanity_check();
#endif
                }
                else
                {
                  // Total domination
                  // Re-use current and create a new segment
                  Segment<T> *new_seg = 
                    new Segment<T>((*curr_it)->a1, (*next_it)->a1,
                                   current_line, (*curr_it)->dir);
                  new_seg->add_reference();
                  segments.insert(new_seg);
                  (*curr_it)->move_adjacent(new_seg, (*curr_it)->a1);
                  (*curr_it)->move_degenerate(new_seg, 
                                              (*curr_it)->a1, (*next_it)->a1);
                  (*curr_it)->filter_degenerate((*next_it)->a1,
                                                (*next_it)->a2, other_segs);
                  Segment<T> *left_edge = 
                    (*next_it)->find_one_adjacent((*next_it)->a1);
                  Segment<T> *right_edge = 
                    (*next_it)->find_one_adjacent((*next_it)->a2);
                  (*curr_it)->a1 = (*next_it)->a2;
                  left_edge->a1 = current_line;
                  right_edge->a1 = current_line;
                  left_edge->clear_adjacent(left_edge->a1);
                  left_edge->replace_adjacent((*next_it), new_seg);
                  new_seg->add_adjacent(left_edge);
                  right_edge->clear_adjacent(right_edge->a1);
                  right_edge->replace_adjacent((*next_it), (*curr_it));
                  (*curr_it)->clear_adjacent((*curr_it)->a1);
                  (*curr_it)->add_adjacent(right_edge);
#ifdef DEBUG_HIGH_LEVEL
                  left_edge->sanity_check();
                  right_edge->sanity_check();
                  (*curr_it)->sanity_check();
                  new_seg->sanity_check();
#endif
                }
                // In all of these casese we always delete next
#ifdef DEBUG_HIGH_LEVEL
                assert(segments.find(*next_it) != segments.end());
#endif
                (*next_it)->filter_degenerate(other_segs);
                segments.erase(*next_it);
                if ((*next_it)->remove_reference())
                  delete (*next_it);
                // make sure we don't try to use it again
                to_remove.insert(*next_it);
              }
              // Next case domination by next
              else if (((*curr_it)->a1 >= (*next_it)->a1) &&
                       ((*curr_it)->a2 <= (*next_it)->a2))
              {
                result = true;
#ifdef DEBUG_HIGH_LEVEL
                assert((*curr_it)->dir != (*next_it)->dir);
#endif
                if ((*curr_it)->a1 == (*next_it)->a1)
                {
                  // Left edge is the same
                  Segment<T> *right_edge = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a2);
                  // See if the left edges are adjoining
                  Segment<T> *adj = 
                    (*next_it)->find_adjoining((*curr_it), (*next_it)->a1);
                  (*next_it)->filter_degenerate((*next_it)->a1, 
                                                (*curr_it)->a2, other_segs);
                  right_edge->a2 = next_line;
                  (*next_it)->a1 = (*curr_it)->a2;
                  right_edge->clear_adjacent(right_edge->a2);
                  right_edge->replace_adjacent((*curr_it), (*next_it));
                  if (adj != NULL)
                  {
                    // Only one adjacent left edge, just need to remove it
                    (*next_it)->clear_adjacent((*next_it)->a1);
                    (*next_it)->replace_adjacent(adj, right_edge);
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(adj) != other_segs.end());
#endif
                    adj->filter_degenerate(segments);
                    other_segs.erase(adj);
                    if (adj->remove_reference())
                      delete adj;
                  }
                  else
                  {
                    // Two separate edges that we need to merge
                    Segment<T> *curr_left = 
                      (*curr_it)->find_one_adjacent((*curr_it)->a1);
                    Segment<T> *next_left = 
                      (*next_it)->find_one_adjacent((*next_it)->a1);
#ifdef DEBUG_HIGH_LEVEL
                    assert(curr_left->dir == next_left->dir);
#endif
                    (*next_it)->clear_adjacent((*next_it)->a1);
                    (*next_it)->replace_adjacent(next_left, right_edge);
                    // Keep the current left edge
                    curr_left->a2 = next_left->a2;
                    next_left->move_adjacent(curr_left, next_left->a2);
                    next_left->move_degenerate(curr_left);
                    // Now we can delete next left
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(next_left) != other_segs.end());
#endif
                    other_segs.erase(next_left);
                    if (next_left->remove_reference())
                      delete next_left;
                  }
#ifdef DEBUG_HIGH_LEVEL
                  right_edge->sanity_check();
                  (*next_it)->sanity_check();
#endif
                }
                else if ((*curr_it)->a2 == (*next_it)->a2)
                {
                  // Right edge is the same
                  Segment<T> *left_edge = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a1);
                  // See if they are adjoining or not
                  Segment<T> *adj = 
                    (*curr_it)->find_adjoining((*next_it), (*curr_it)->a2);
                  (*next_it)->filter_degenerate((*curr_it)->a1, 
                                                (*next_it)->a2, other_segs);
                  left_edge->a2 = next_line;
                  (*next_it)->a2 = (*curr_it)->a1;
                  left_edge->clear_adjacent(left_edge->a2);
                  left_edge->replace_adjacent((*curr_it), (*next_it));
                  if (adj != NULL)
                  {
                    (*next_it)->clear_adjacent((*next_it)->a2);
                    (*next_it)->replace_adjacent(adj, left_edge);
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(adj) != other_segs.end());
#endif
                    adj->filter_degenerate(segments);
                    other_segs.erase(adj);
                    if (adj->remove_reference())
                      delete adj;
                  }
                  else
                  {
                    // Two separate segments that we need to merge
                    Segment<T> *curr_right = 
                      (*curr_it)->find_one_adjacent((*curr_it)->a2);
                    Segment<T> *next_right = 
                      (*next_it)->find_one_adjacent((*next_it)->a2);
#ifdef DEBUG_HIGH_LEVEL
                    assert(curr_right->dir == next_right->dir);
#endif
                    (*next_it)->clear_adjacent((*next_it)->a2);
                    (*next_it)->replace_adjacent(next_right, left_edge);
                    // Keep the current right edge
                    curr_right->a2 = next_right->a2;
                    next_right->move_adjacent(curr_right, next_right->a2);
                    next_right->move_degenerate(curr_right);
                    // Now we can delete the next right
#ifdef DEBUG_HIGH_LEVEL
                    assert(other_segs.find(next_right) != other_segs.end());
#endif
                    other_segs.erase(next_right);
                    if (next_right->remove_reference())
                      delete next_right;
                  }
#ifdef DEBUG_HIGH_LEVEL
                  left_edge->sanity_check();
                  (*next_it)->sanity_check();
#endif
                }
                else
                {
                  // Total domination by next
                  Segment<T> *new_seg = 
                    new Segment<T>((*curr_it)->a2, (*next_it)->a2,
                                   next_line, (*next_it)->dir);
                  new_seg->add_reference();
                  segments.insert(new_seg);
                  (*next_it)->move_adjacent(new_seg, (*next_it)->a2);
                  (*next_it)->move_degenerate(new_seg,
                                              (*curr_it)->a2, (*next_it)->a2);
                  (*next_it)->filter_degenerate((*curr_it)->a1,
                                                (*curr_it)->a2, other_segs);
                  Segment<T> *left_edge = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a1);
                  Segment<T> *right_edge = 
                    (*curr_it)->find_one_adjacent((*curr_it)->a2);
                  (*next_it)->a2 = (*curr_it)->a1;
                  left_edge->a2 = next_line;
                  right_edge->a2 = next_line;
                  left_edge->clear_adjacent(left_edge->a2);
                  left_edge->replace_adjacent((*curr_it), (*next_it));
                  (*next_it)->clear_adjacent((*next_it)->a2);
                  (*next_it)->add_adjacent(left_edge);
                  right_edge->clear_adjacent(right_edge->a2);
                  right_edge->replace_adjacent((*curr_it), new_seg);
                  new_seg->add_adjacent(right_edge);
#ifdef DEBUG_HIGH_LEVEL
                  left_edge->sanity_check();
                  right_edge->sanity_check();
                  (*next_it)->sanity_check();
                  new_seg->sanity_check();
#endif
                  // Add the new segment to the set to be added
                  to_add.push_back(new_seg);
                }
                // We always delete current in all cases
#ifdef DEBUG_HIGH_LEVEL
                assert(segments.find(*curr_it) != segments.end());
#endif
                (*curr_it)->filter_degenerate(other_segs);
                segments.erase(*curr_it);
                if ((*curr_it)->remove_reference())
                  delete (*curr_it);
                // Since we were completely dominated by next
                // we can break out of the inner loop
                break;
              }
              // Strict left overlap by current/right overlap by next
              else if (((*curr_it)->a1 < (*next_it)->a1) &&
                       ((*curr_it)->a2 > (*next_it)->a1))
              {
                result = true;
#ifdef DEBUG_HIGH_LEVEL
                assert((*curr_it)->dir != (*next_it)->dir);
#endif
                Segment<T> *left_edge = 
                  (*next_it)->find_one_adjacent((*next_it)->a1);
                Segment<T> *right_edge = 
                  (*curr_it)->find_one_adjacent((*curr_it)->a2);
                (*curr_it)->filter_degenerate((*next_it)->a1,
                                              (*curr_it)->a2, other_segs);
                (*next_it)->filter_degenerate((*next_it)->a1,
                                              (*curr_it)->a2, other_segs);
                left_edge->a1 = current_line;
                right_edge->a2 = next_line;
                T temp = (*curr_it)->a2;
                (*curr_it)->a2 = (*next_it)->a1;
                (*next_it)->a1 = temp;
                left_edge->clear_adjacent(left_edge->a1);
                left_edge->replace_adjacent((*next_it), (*curr_it));
                right_edge->clear_adjacent(right_edge->a2);
                right_edge->replace_adjacent((*curr_it), (*next_it));
                (*next_it)->clear_adjacent((*next_it)->a1);
                (*next_it)->replace_adjacent(left_edge, right_edge);
                (*curr_it)->clear_adjacent((*curr_it)->a2);
                (*curr_it)->replace_adjacent(right_edge, left_edge);
#ifdef DEBUG_HIGH_LEVEL
                left_edge->sanity_check();
                right_edge->sanity_check();
                (*next_it)->sanity_check();
                (*curr_it)->sanity_check();
#endif
              }
              // Strict left overlap by next/right overlap by current
              else if (((*curr_it)->a1 > (*next_it)->a1) &&
                       ((*curr_it)->a1 < (*next_it)->a2))
              {
                result = true;
#ifdef DEBUG_HIGH_LEVEL
                assert((*curr_it)->dir != (*next_it)->dir);
#endif
                Segment<T> *left_edge = 
                  (*curr_it)->find_one_adjacent((*curr_it)->a1);
                Segment<T> *right_edge = 
                  (*next_it)->find_one_adjacent((*next_it)->a2);
                (*curr_it)->filter_degenerate((*curr_it)->a1,
                                              (*next_it)->a2, other_segs);
                (*next_it)->filter_degenerate((*curr_it)->a1,
                                              (*next_it)->a2, other_segs);
                left_edge->a2 = next_line;
                right_edge->a1 = current_line;
                T temp = (*curr_it)->a1;
                (*curr_it)->a1 = (*next_it)->a2;
                (*next_it)->a2 = temp;
                left_edge->clear_adjacent(left_edge->a2);
                left_edge->replace_adjacent((*curr_it), (*next_it));
                right_edge->clear_adjacent(right_edge->a1);
                right_edge->replace_adjacent((*next_it), (*curr_it));
                (*curr_it)->clear_adjacent((*curr_it)->a1);
                (*curr_it)->replace_adjacent(left_edge, right_edge);
                (*next_it)->clear_adjacent((*next_it)->a2);
                (*next_it)->replace_adjacent(right_edge, left_edge);
#ifdef DEBUG_HIGH_LEVEL
                left_edge->sanity_check();
                right_edge->sanity_check();
                (*next_it)->sanity_check();
                (*curr_it)->sanity_check();
#endif
              }
              // Otherwise they touch on the ends and we don't care
            }
            // Before going onto the next loop we have to add
            // any new next edges to the next segments set 
            if (!to_add.empty())
            {
              next_segments.insert(next_segments.end(),
                                   to_add.begin(), to_add.end());
              to_add.clear();
            }
          }
          current_line = next_line;
          current_segments.clear();
          for (typename std::vector<Segment<T>*>::const_iterator next_it =
                next_segments.begin(); next_it != 
                next_segments.end(); next_it++)
          {
            if (to_remove.find(*next_it) == to_remove.end())
            {
#ifdef DEBUG_HIGH_LEVEL
              (*next_it)->sanity_check();
#endif
              current_segments.push_back(*next_it);
            }
          }
        }
        else
        {
          current_line = next_line;
          current_segments = next_segments;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ Segment<T>* RectangleSet<T,DISCRETE>::find_low(
                 const Segment<T> &segment, const std::set<Segment<T>*> &bounds)
    //--------------------------------------------------------------------------
    {
      Segment<T> *result = NULL;
      T diff;
      for (typename std::set<Segment<T>*>::const_iterator it = bounds.begin();
            it != bounds.end(); it++)
      {
        if (!(*it)->touches(segment))
          continue;
        T distance = segment.distance_low(*(*it));
        if (distance < 0)
          continue;
        if ((result == NULL) || (distance < diff))
        {
          result = (*it);
          diff = distance;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ Segment<T>* RectangleSet<T,DISCRETE>::find_high(
                 const Segment<T> &segment, const std::set<Segment<T>*> &bounds)
    //--------------------------------------------------------------------------
    {
      Segment<T> *result = NULL;
      T diff;
      for (typename std::set<Segment<T>*>::const_iterator it = bounds.begin();
            it != bounds.end(); it++)
      {
        if (!(*it)->touches(segment))
          continue;
        T distance = segment.distance_high(*(*it));
        if (distance > 0)
          continue;
        if ((result == NULL) || (distance > diff))
        {
          result = (*it);
          diff = distance;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ bool RectangleSet<T,DISCRETE>::has_divisor(
                 const Segment<T> &segment, const std::set<Segment<T>*> &bounds)
    //--------------------------------------------------------------------------
    {
      for (typename std::set<Segment<T>*>::const_iterator it = 
            bounds.begin(); it != bounds.end(); it++)
      {
        if ((*it)->divides(segment))
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ void RectangleSet<T,DISCRETE>::boundary_edges(
        const std::set<Segment<T>*> &xs, const std::set<Segment<T>*> &xs_prime,
        const std::set<Segment<T>*> &ys_prime, std::set<Segment<T>*> &result)
    //--------------------------------------------------------------------------
    {
      std::vector<SplitSegment<T> > split_segments;
      for (typename std::set<Segment<T>*>::const_iterator it = 
            xs.begin(); it != xs.end(); it++)
      {
        split_segment((*it), ys_prime, split_segments);
      }
      for (typename std::vector<SplitSegment<T> >::const_iterator it =
            split_segments.begin(); it != split_segments.end(); it++)
      {
        if (boundary(*it))
          result.insert(it->segment);
        else if (it->segment->remove_reference())
          delete it->segment;
      }
    }
    
    // Small helper class for comparing segments
    template<typename T>
    class SegmentComparator {
    public:
      bool operator()(const Segment<T> *left, const Segment<T> *right)
      {
        if (left->b < right->b)
          return true;
        else if (left->b > right->b)
          return false;
        else
        {
          if (left->a1 < right->a1)
            return true;
          else if (left->a1 > right->a1)
            return false;
          else
          {
            // Sort in reverse order here
            if (left->a2 > right->a2)
              return true;
            else if (left->a2 < right->a2)
              return false;
            else
            {
              if (left->dir < right->dir)
                return true;
              else if (left->dir > right->dir)
                return false;
              else
                return (left < right);
            }
          }
        }
      }
    };

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ void RectangleSet<T,DISCRETE>::merge_segments(
                                                std::set<Segment<T>*> &segments)
    //--------------------------------------------------------------------------
    {
      std::set<Segment<T>*,SegmentComparator<T> > sorted_segments(
                                              segments.begin(), segments.end());
#ifdef DEBUG_HIGH_LEVEL
      assert(sorted_segments.size() == segments.size());
#endif
      Segment<T> *current = NULL;
      for (typename std::set<Segment<T>*,SegmentComparator<T> >::const_iterator
            it = sorted_segments.begin(); it != sorted_segments.end(); it++)
      {
        if (current == NULL)
          current = (*it);
        else
        {
          if ((*it)->b != current->b)
            current = (*it);
          else if (((*it)->a1 < current->a2) &&
                    (*it)->points_none() && current->points_none())
          {
            (*it)->a1 = current->a1;
#ifdef DEBUG_HIGH_LEVEL
            assert(segments.find(current) != segments.end());
#endif
            segments.erase(current);
            if (current->remove_reference())
              delete current;
            current = (*it);
          }
          else if ((*it)->a2 <= current->a2)
          {
#ifdef DEBUG_HIGH_LEVEL
            assert((*it)->points_none() ||
                   (*it)->dir == current->dir);
            assert(segments.find(*it) != segments.end());
#endif
            segments.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
            // Keep current
          }
          else if (((*it)->a1 == current->a2) && 
                   ((*it)->dir == current->dir) && !current->points_none())
          {
            current->a2 = (*it)->a2;
#ifdef DEBUG_HIGH_LEVEL
            assert(segments.find(*it) != segments.end());
#endif
            segments.erase(*it);
            if ((*it)->remove_reference())
              delete (*it);
            // Keep current
          }
          else
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(((*it)->a1 > current->a2) ||
                   (((current->a2 == (*it)->a1) &&
                     ((current->dir != (*it)->dir) ||
                      current->points_none()))));
#endif
            // Just update current
            current = (*it);
          }
        }
      }
    }

    template<typename T>
    struct SplitComparator {
    public:
      bool operator()(const Segment<T> *left, const Segment<T> *right)
      {
        // Since there should only be one of splitter across
        // a given segment at each 'b' value we don't need
        // to bother checking for less than on other dimensions
        return (left->b < right->b);
      }
    };

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ void RectangleSet<T,DISCRETE>::split_segment(Segment<T> *segment,
        const std::set<Segment<T>*> &ys, std::vector<SplitSegment<T> > &splits)
    //--------------------------------------------------------------------------
    {
      std::set<Segment<T>*,SplitComparator<T> > splitters;
      for (typename std::set<Segment<T>*>::const_iterator it = ys.begin();
            it != ys.end(); it++)
      {
        if ((*it)->divides(*segment))
          splitters.insert(*it);
      }
      Segment<T> *low = find_low(*segment, ys);
      Segment<T> *high = find_high(*segment, ys);
      if (splitters.empty())
      {
        splits.push_back(SplitSegment<T>(segment, low, high));
        // Add a reference to the segment that we added
        segment->add_reference();
      }
      else
      {
        typename std::set<Segment<T>*>::const_iterator it = splitters.begin();
        Segment<T> *first = new Segment<T>(segment->a1, (*it)->b, segment->b, 
                                           segment->dir);
        first->add_reference();
        splits.push_back(SplitSegment<T>(first, low, (*it)));
        Segment<T> *previous = (*it);
        it++;
        while (it != splitters.end())
        {
          Segment<T> *next = new Segment<T>(previous->b, (*it)->b, segment->b,
                                            segment->dir);
          next->add_reference();
          splits.push_back(SplitSegment<T>(next, previous, (*it)));
          previous = (*it);
          it++;
        }
        Segment<T> *last = new Segment<T>(previous->b, segment->a2, segment->b,
                                          segment->dir);
        last->add_reference();
        splits.push_back(SplitSegment<T>(last, previous, high));
      }
    }
    
    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ bool RectangleSet<T,DISCRETE>::boundary(
                                                   const SplitSegment<T> &split)
    //--------------------------------------------------------------------------
    {
      if (split.lower == NULL)
        return true;
      if (split.higher == NULL)
        return true;
      Segment<T> *adjoining = split.lower->find_adjoining(split.higher,
                                                      split.segment->b);
      if ((adjoining != NULL) &&
          (adjoining->a1 <= split.segment->a1) &&
          (adjoining->a2 >= split.segment->a2))
      {
        return ((split.segment->dir == adjoining->dir) ||
                split.segment->points_none() ||
                adjoining->points_none());
      }
      else
      {
        return split.lower->points_none() || split.lower->points_left();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    /*static*/ bool RectangleSet<T,DISCRETE>::has_overlap(Segment<T> *segment,
                                            const std::set<Segment<T>*> &bounds)
    //--------------------------------------------------------------------------
    {
      for (typename std::set<Segment<T>*>::const_iterator it = 
            bounds.begin(); it != bounds.end(); it++)
      {
        if (!(*it)->points_none())
          continue;
        if ((*it)->b != segment->b)
          continue;
        if (((*it)->a1 <= segment->a1) &&
            (segment->a2 <= (*it)->a2))
          return true;
      }
      return false;
    } 
  };
};

#endif // __LEGION_RECTANGLE_SET_H__

// EOF


