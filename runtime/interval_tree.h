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


#ifndef __LEGION_INTERVAL_TREE_H__
#define __LEGION_INTERVAL_TREE_H__

#include <cassert>
#include <cstdlib>

namespace LegionRuntime {
  namespace HighLevel {

    /**
     * \class IntervalTreeNode
     * A class for representing nodes in an interval
     * tree. If it is an intermediate node then it 
     * represents a whole in the tree, otherwise if
     * it is a leaf then it represents an actual interval.
     */
    template<typename T, bool DISCRETE>
    class IntervalTreeNode {
    public:
      IntervalTreeNode(T left, T right);
      IntervalTreeNode(const IntervalTreeNode &rhs);
      ~IntervalTreeNode(void);
    public:
      IntervalTreeNode& operator=(const IntervalTreeNode &rhs);
    public:
      IntervalTreeNode<T,DISCRETE>* insert(T left, T right);
      IntervalTreeNode<T,DISCRETE>* insert_local(T left, T right);
      bool intersects(T left, T right) const;
      bool dominates(T left, T right) const;
      int count(void) const;
      IntervalTreeNode<T,DISCRETE>* merge(void);
      IntervalTreeNode<T,DISCRETE>* reinsert(
                            IntervalTreeNode<T,DISCRETE> *target);
    public:
      void sanity_check(void);
    private:
      T left_bound, right_bound;
      IntervalTreeNode<T, DISCRETE> *left_node, *right_node;
    };
    
    /** 
     * \class IntervalTree
     * A slightly modified version of interval tree
     * that collapses intervals that overlap into single
     * intervals to help with testing for intersection
     * and domination.
     */
    template<typename T, bool DISCRETE>
    class IntervalTree {
    public:
      IntervalTree(void);
      IntervalTree(const IntervalTree &rhs);
      ~IntervalTree(void);
    public:
      IntervalTree& operator=(const IntervalTree &rhs);
    public:
      void insert(T left, T right);
      bool intersects(T left, T right) const;
      bool dominates(T left, T right) const;
    private:
      IntervalTreeNode<T, DISCRETE> *root;
    };

    /////////////////////////////////////////////////////////////
    // Interval Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>::IntervalTreeNode(T left, T right)
      : left_bound(left), right_bound(right), left_node(NULL), right_node(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>::IntervalTreeNode(const IntervalTreeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>::~IntervalTreeNode(void)
    //--------------------------------------------------------------------------
    {
      if (left_node != NULL)
        delete left_node;
      if (right_node != NULL)
        delete right_node;
      left_node = NULL;
      right_node = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>& IntervalTreeNode<T,DISCRETE>::operator=(
                                                    const IntervalTreeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>* IntervalTreeNode<T,DISCRETE>::insert(
                                                                T left, T right)
    //--------------------------------------------------------------------------
    {
      // Six cases here
      // 1. Dominates
      // 2. Contained within
      // 3. Overlap to the left
      // 4. Overlap to the right
      // 5. All to the left
      // 6. All to the right
      if ((left_bound <= left) && (right <= right_bound))
      {
        if (left_node != NULL)
        {
          // Insert in the side with fewer segments
#ifdef DEBUG_HIGH_LEVEL
          assert(right_node != NULL);
#endif
          return insert_local(left, right);
        }
        // Otherwise we are a base segment and since we
        // dominate then we are done
      }
      else if ((left <= left_bound) && (right_bound <= right))
      {
        // No need to keep sub-segments anymore, we can delete them
        if (left_node != NULL)
        {
          delete left_node;
          left_node = NULL;
        }
        if (right_node != NULL)
        {
          delete right_node;
          right_node = NULL;
        }
        left_bound = left;
        right_bound = right;
      }
      else if ((left_bound <= right) && (right <= right_bound))
      {
        left_bound = left;
        if (left_node != NULL)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(right_node != NULL);
#endif
          return insert_local(left, right);
        }
      }
      else if ((left_bound <= left) && (left <= right_bound))
      {
        right_bound = right;
        if (left_node != NULL)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(right_node != NULL);
#endif
          return insert_local(left, right);
        }
      }
      else if (DISCRETE && (left == (right_bound+1)))
      {
        if (right_node != NULL)
          right_node = right_node->insert(left, right);
        right_bound = right;
      }
      else if (DISCRETE && ((right+1) == left_bound))
      {
        if (left_node != NULL)
          left_node = left_node->insert(left, right);
        left_bound = left;
      }
      else if (left > right_bound)
      {
        if (right_node != NULL)
        {
          right_node = right_node->insert(left, right);
        }
        else
        {
          // Otherwise make ourselves a new intermedite node
          left_node = 
            new IntervalTreeNode<T,DISCRETE>(left_bound, right_bound);
          right_node = 
            new IntervalTreeNode<T,DISCRETE>(left, right);
        }
        right_bound = right;
      }
      else if (right < left_bound)
      {
        if (left_node != NULL)
        {
          left_node = left_node->insert(left, right);
        }
        else
        {
          left_node = 
            new IntervalTreeNode<T,DISCRETE>(left, right);
          right_node = 
            new IntervalTreeNode<T,DISCRETE>(left_bound, right_bound);
        }
        left_bound = left;
      }
      else
        assert(false); // should never hit this case
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
      return this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>* IntervalTreeNode<T,DISCRETE>::insert_local(
                                                                T left, T right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left_node != NULL);
      assert(right_node != NULL);
#endif
      bool intersect_left = left_node->intersects(left, right);
      bool intersect_right = right_node->intersects(left, right);
      if (intersect_left && intersect_right)
      {
        left_node = left_node->insert(left, right);
        IntervalTreeNode<T,DISCRETE> *result = right_node->reinsert(left_node);
        // make sure we don't delete the new tree
        left_node = NULL;
        delete this;
#ifdef DEBUG_HIGH_LEVEL
        result->sanity_check();
#endif
        return result;
      }
      else if (intersect_left)
        left_node = left_node->insert(left, right);
      else if (intersect_right)
        right_node = right_node->insert(left, right);
      else
      {
        int count_left = left_node->count();
        int count_right = right_node->count();
        if (count_left <= count_right)
          left_node = left_node->insert(left, right);
        else
          right_node = right_node->insert(left, right);
      }
#ifdef DEBUG_HIGH_LEVEL
      sanity_check();
#endif
      return this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    bool IntervalTreeNode<T,DISCRETE>::intersects(T left, T right) const
    //--------------------------------------------------------------------------
    {
      if (right < left_bound)
        return false;
      if (left > right_bound)
        return false;
      if (left_node != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(right_node != NULL);
#endif
        if (left_node->intersects(left, right))
          return true;
        if (right_node->intersects(left, right))
          return true;
        return false;
      }
      else
      {
        if ((left <= left_bound) && (right >= right_bound))
          return true;
        if ((left_bound <= left) && (left <= right_bound))
          return true;
        if ((left_bound <= right) && (right <= right_bound))
          return true;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    bool IntervalTreeNode<T,DISCRETE>::dominates(T left, T right) const
    //--------------------------------------------------------------------------
    {
      // See if we dominate it
      if ((left_bound <= left) && (right <= right_bound))
      {
        // If we are an intermediate node, we need to see
        // if any of our children dominate it
        if (left_node != NULL)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(right_node != NULL);
#endif
          if (left_node->dominates(left, right))
            return true;
          if (right_node->dominates(left, right))
            return true;
        }
        else
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    int IntervalTreeNode<T,DISCRETE>::count(void) const
    //--------------------------------------------------------------------------
    {
      int result = 1;
      if (left_node != NULL)
        result += left_node->count();
      if (right_node != NULL)
        result += right_node->count();
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>* IntervalTreeNode<T,DISCRETE>::merge(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(((left_node == NULL) && (right_node == NULL)) ||
             ((left_node != NULL) && (right_node != NULL)));
#endif
      if (left_node == NULL)
        return this;
      if (DISCRETE)
      {
        if ((left_node->right_bound+1) == right_node->left_bound)
        {
          IntervalTreeNode<T,DISCRETE> *result = 
                              right_node->reinsert(left_node);
          left_node = NULL;
          delete this;
#ifdef DEBUG_HIGH_LEVEL
          result->sanity_check();
#endif
          return result;
        }
        else
        {
          left_node = left_node->merge();
          right_node = right_node->merge();
          // Check again
          if ((left_node->right_bound+1) == right_node->left_bound)
          {
            IntervalTreeNode<T,DISCRETE> *result = 
              right_node->reinsert(left_node);
            left_node = NULL;
            delete this;
#ifdef DEBUG_HIGH_LEVEL
            result->sanity_check();
#endif
            return result;
          }
        }
      }
      else
      {
        if (left_node->right_bound == right_node->left_bound)  
        {
          IntervalTreeNode<T,DISCRETE> *result = 
                              right_node->reinsert(left_node);
          left_node = NULL;
          delete this;
#ifdef DEBUG_HIGH_LEVEL
          result->sanity_check();
#endif
          return result;
        }
        else
        {
          left_node = left_node->merge();
          right_node = right_node->merge();
          // Check again
          if (left_node->right_bound == right_node->left_bound)
          {
            IntervalTreeNode<T,DISCRETE> *result = 
                              right_node->reinsert(left_node);
            left_node = NULL;
            delete this;
#ifdef DEBUG_HIGH_LEVEL
            result->sanity_check();
#endif
            return result;
          }
        }
      }
      return this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTreeNode<T,DISCRETE>* IntervalTreeNode<T,DISCRETE>::reinsert(
                                           IntervalTreeNode<T,DISCRETE> *target)
    //--------------------------------------------------------------------------
    {
      if (left_node != NULL)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(right_node != NULL);
#endif
        target = left_node->reinsert(target);
        target = right_node->reinsert(target);
        return target;
      }
      else
      {
        return target->insert(left_bound, right_bound);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    void IntervalTreeNode<T,DISCRETE>::sanity_check(void)
    //--------------------------------------------------------------------------
    {
      if (left_node != NULL)
      {
        assert(right_node != NULL);
        assert(left_node->right_bound < right_bound);
        assert(right_node->left_bound > left_bound);
        assert(left_node->right_bound <= left_node->right_bound);
      }
    }

    /////////////////////////////////////////////////////////////
    // Interval Tree 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTree<T,DISCRETE>::IntervalTree(void)
      : root(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTree<T,DISCRETE>::IntervalTree(const IntervalTree &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTree<T,DISCRETE>::~IntervalTree(void)
    //--------------------------------------------------------------------------
    {
      if (root != NULL)
        delete root;
      root = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    IntervalTree<T,DISCRETE>& IntervalTree<T,DISCRETE>::operator=(
                                                        const IntervalTree &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    void IntervalTree<T,DISCRETE>::insert(T left, T right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left <= right);
#endif
      if (root == NULL)
        root = new IntervalTreeNode<T,DISCRETE>(left, right);
      else
      {
        root = root->insert(left, right);
        root = root->merge();
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    bool IntervalTree<T,DISCRETE>::intersects(T left, T right) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left <= right);
#endif
      if (root == NULL)
        return false;
      return root->intersects(left, right);
    }

    //--------------------------------------------------------------------------
    template<typename T, bool DISCRETE>
    bool IntervalTree<T,DISCRETE>::dominates(T left, T right) const
    //--------------------------------------------------------------------------
    {
      if (root == NULL)
        return false;
      return root->dominates(left, right);
    }

  };
};

#endif // __LEGION_INTERVAL_TREE_H__

// EOF

