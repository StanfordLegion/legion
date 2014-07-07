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

namespace LegionRuntime {
  namespace HighLevel {

    /**
     * \class IntervalTreeNode
     * A class for representing nodes in an interval
     * tree. If it is an intermediate node then it 
     * represents a whole in the tree, otherwise if
     * it is a leaf then it represents an actual interval.
     */
    template<typename T>
    class IntervalTreeNode {
    public:
      IntervalTreeNode(T left, T right);
      IntervalTreeNode(const IntervalTreeNode &rhs);
      ~IntervalTreeNode(void);
    public:
      IntervalTreeNode& operator=(const IntervalTreeNode &rhs);
    public:
      IntervalTreeNode<T>* insert(T left, T right);
      bool intersects(T left, T right) const;
      bool dominates(T left, T right) const;
      int count(void) const;
      IntervalTreeNode<T>* reinsert(IntervalTreeNode<T> *target) const;
    private:
      bool intermediate;
      T left_bound, right_bound;
      IntervalTreeNode<T> *left_node, *right_node;
    };
    
    /** 
     * \class IntervalTree
     * A slightly modified version of interval tree
     * that collapses intervals that overlap into single
     * intervals to help with testing for intersection
     * and domination.
     */
    template<typename T>
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
      IntervalTreeNode<T> *root;
    };

    /////////////////////////////////////////////////////////////
    // Interval Tree Node 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTreeNode<T>::IntervalTreeNode(T left, T right)
      : intermediate(false), left_bound(left), right_bound(right), 
        left_node(NULL), right_node(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTreeNode<T>::IntervalTreeNode(const IntervalTreeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTreeNode<T>::~IntervalTreeNode(void)
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
    template<typename T>
    IntervalTreeNode<T>& IntervalTreeNode<T>::operator=(
                                                    const IntervalTreeNode &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTreeNode<T>* IntervalTreeNode<T>::insert(T left, T right)
    //--------------------------------------------------------------------------
    {
      if (intermediate)
      {
        // Six cases here
        // 1. All to the left
        // 2. All to the right
        // 3. Overlap to the left
        // 4. Overlap to the right
        // 5. Contained within
        // 6. Dominates
        if (right < left_bound)
          left_node = left_node->insert(left, right);
        else if (left > right_bound)
          right_node = right_node->insert(left, right);
        else if ((left > left_bound) && (right < right_bound))
        {
          // contained within
          int left_size = left_node->count();
          int right_size = right_node->count();
          // Decide whether to insert left or right
          if (left_size <= right_size)
          {
            left_bound = right;
            left_node = left_node->insert(left, right);
          }
          else
          {
            right_bound = left;
            right_node = right_node->insert(left, right);
          }
        }
        else if ((left <= left_bound) && (right >= right_bound))
        {
          // dominates
          int left_size = left_node->count();
          int right_size = right_node->count();
          if (left_size <= right_size)
          {
            // insert in the right side
            // reinsert all the nodes in the left in the right
            // return the right side after deleting ourselves
            IntervalTreeNode<T> *result = right_node->insert(left, right);
            result = left_node->reinsert(result);
            // make sure we don't delete the new tree
            right_node = NULL;
            delete this;
            return result;
          }
          else
          {
            IntervalTreeNode<T> *result = left_node->insert(left, right);
            result = right_node->reinsert(result);
            // make sure we don't delete the new tree
            left_node = NULL;
            delete this;
            return result;
          }
        }
        else if ((left <= left_bound) && (right < right_bound))
        {
          // overlap to the left
          left_bound = right;
          left_node = left_node->insert(left, right);
        }
        else if ((right >= right_bound) && (left > left_bound))
        {
          // overlap to the right
          right_bound = left;
          right_node = right_node->insert(left, right);
        }
        else
          assert(false); // should never get here
      }
      else
      {
        // check to see if we should merge
        bool merged = false;
        if ((left < left_bound) && (right >= left_bound))
        {
          left_bound = left;
          merged = true;
        }
        if ((right > right_bound) && (left <= right_bound))
        {
          right_bound = right;
          merged = true;
        }
        if (!merged)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert((left_bound > right) || (right_bound < left));
#endif
          // Make a new node for ourself and our new interval
          IntervalTreeNode<T> *self = 
                            new IntervalTreeNode<T>(left_bound, right_bound);
          IntervalTreeNode<T> *other = new IntervalTreeNode<T>(left, right);
          // Convert ourselves into an intermediate node
          intermediate = true; 
          if (left_bound > right)
          {
            left_node = other;
            right_node = self;
            right_bound = left_bound;
            left_bound = right;
          }
          else
          {
            left_node = self;
            right_node = other;
            left_bound = right_bound;
            right_bound = left;
          }
        }
      }
      return this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool IntervalTreeNode<T>::intersects(T left, T right) const
    //--------------------------------------------------------------------------
    {
      if (intermediate)
      {
        if (right <= left_bound)
          return left->intersects(left, right);
        if (left >= right_bound)
          return right->intersects(left, right);
        return false;
      }
      else
      {
        if ((left >= left_bound) && (left <= right_bound))
          return true;
        if ((right >= left_bound) && (right <= right_bound))
          return true;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool IntervalTreeNode<T>::dominates(T left, T right) const
    //--------------------------------------------------------------------------
    {
      if (intermediate)
      {
        if (right <= left_bound)
          return left_node->dominates(left, right);
        if (left >= right_bound)
          return right_node->dominates(left, right);
        return false;
      }
      else
      {
        if ((left >= left_bound) && (right <= right_bound))
          return true;
        return false;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    int IntervalTreeNode<T>::count(void) const
    //--------------------------------------------------------------------------
    {
      if (intermediate)
        return (left_node->count() + right_node->count());
      else
        return 1;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTreeNode<T>* IntervalTreeNode<T>::reinsert(
                                              IntervalTreeNode<T> *target) const
    //--------------------------------------------------------------------------
    {
      if (intermediate)
      {
        target = left_node->reinsert(target);
        target = right_node->reinsert(target);
        return target;
      }
      else
        return target->insert(left_bound, right_bound);
    }

    /////////////////////////////////////////////////////////////
    // Interval Tree 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTree<T>::IntervalTree(void)
      : root(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTree<T>::IntervalTree(const IntervalTree &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTree<T>::~IntervalTree(void)
    //--------------------------------------------------------------------------
    {
      if (root != NULL)
        delete root;
      root = NULL;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    IntervalTree<T>& IntervalTree<T>::operator=(const IntervalTree &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void IntervalTree<T>::insert(T left, T right)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left <= right);
#endif
      if (root == NULL)
        root = new IntervalTreeNode<T>(left, right);
      else
        root = root->insert(left, right);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool IntervalTree<T>::intersects(T left, T right) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(left <= right);
#endif
      if (root = NULL)
        return false;
      return root->intersects(left, right);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool IntervalTree<T>::dominates(T left, T right) const
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
