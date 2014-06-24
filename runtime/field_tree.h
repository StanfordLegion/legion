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


#ifndef __LEGION_FIELD_TREE_H__
#define __LEGION_FIELD_TREE_H__

#include "legion_types.h"

namespace LegionRuntime {
  namespace HighLevel {

    /**
     * \class FieldTree
     * This is a template class for supporting
     * analysis on user types with field masks.
     * It aids in minimizing the number of analyses
     * that need to be performed based on field
     * independence tests. It must support a 
     * user type UT which has a field called
     * 'field_mask' that is publically visible.
     * It must also be instantiated with an
     * analysis type AT which supports a method
     * analyze on user type UT.
     */
    template<typename UT>
    class FieldTree {
    public:
      FieldTree(const FieldMask &mask, bool merge_node = false);
      FieldTree(const FieldTree<UT> &rhs);
      ~FieldTree(void);
    public:
      FieldTree<UT>& operator=(const FieldTree<UT> &rhs);
    public:
      template<typename AT>
      inline void analyze(const FieldMask &mask, AT &analyzer);
      inline void insert(const UT &user, bool recurse = true);
    public:
      void pack_field_tree(Serializer &rez);
      void unpack_field_tree(Deserializer &derez);
    private:
      template<typename AT>
      void analyze_recurse(const FieldMask &mask, AT &analyzer);
      template<typename AT>
      void analyze_no_checks(AT &analyzer);
      template<typename AT>
      void analyze_single(unsigned index, AT &analyzer);
    private:
      template<typename AT>
      inline void analyze_precise(AT &analyzer);
      template<typename AT, bool CHECK>
      inline void analyze_imprecise(const FieldMask &mask, AT &analyzer);
      template<typename AT, bool CHECK>
      inline void analyze_single(const FieldMask &mask, AT &analyzer);
      template<typename AT>
      inline void analyze_single_precise(unsigned index, AT &analyzer);
      template<typename AT>
      inline void analyze_single_imprecise(unsigned index, AT &analyzer);
      template<typename AT>
      inline void analyze_single_single(unsigned index, AT &analyzer);
    private:
      void insert_recurse(const UT &user);
      void insert_single(unsigned index, const UT &user);
    private:
      inline FieldTree<UT>* merge_dominators(
          const std::set<FieldTree<UT>*> &dominators,
          const FieldMask &dominator_mask);
      inline void subsume_node(FieldTree<UT> *dominator_node);
    private:
      inline void add_user(const UT &user, bool precise);
      inline void add_single_user(unsigned index, const UT &user);
      inline void add_child(FieldTree<UT> *child_node);
      void check_state(void);
    public:
      // Not constant so we can mutate it when packing/unpacking
      FieldMask local_mask;
      const bool merge_node;
      const bool single_node;
      // If a single node then this is the index of the set field
      const unsigned single_index;
    private:
      std::set<FieldTree<UT>*> children;
      std::set<FieldTree<UT>*> single_children;
    private:
      std::list<UT> precise_users;
      std::list<UT> imprecise_users;
      std::list<std::pair<UT,unsigned> > single_users;
    };

    // Since the field tree class is templated we have to put the 
    // methods here in the header file to make sure they get
    // instantiated properly (C++ is dumb).
    
    //--------------------------------------------------------------------------
    template<typename UT>
    FieldTree<UT>::FieldTree(const FieldMask &mask, bool merge/* = false*/)
      : local_mask(mask), merge_node(merge),
        single_node(FieldMask::pop_count(mask) == 1), 
        single_index(single_node ? mask.find_first_set() : 0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    FieldTree<UT>::FieldTree(const FieldTree<UT> &rhs)
      : local_mask(FieldMask()), merge_node(false), single_node(false), 
        single_index(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    FieldTree<UT>::~FieldTree(void)
    //--------------------------------------------------------------------------
    {
      // If we're a single node, then there is nothing to do
      if (single_node)
        return;
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        delete (*it);
      }
      children.clear();
      for (typename std::set<FieldTree<UT>*>::const_iterator it =
            single_children.begin(); it != single_children.end(); it++)
      {
        delete (*it);
      }
      single_children.clear();
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    FieldTree<UT>& FieldTree<UT>::operator=(const FieldTree<UT> &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    void FieldTree<UT>::analyze(const FieldMask &mask, AT &analyzer)
    //--------------------------------------------------------------------------
    {
      if (FieldMask::pop_count(mask) == 1)
      {
        unsigned index = mask.find_first_set();
        analyze_single(index, analyzer);
      }
      else
        analyze_recurse(mask, analyzer);
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::insert(const UT &user, bool recurse/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (FieldMask::pop_count(user.field_mask) == 1)
      {
        unsigned index = user.field_mask.find_first_set();
        if (recurse)
          insert_single(index, user);
        else
          add_single_user(index, user);
      }
      else
      {
        if (recurse)
          insert_recurse(user);
        else
          add_user(user, (local_mask == user.field_mask));
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::pack_field_tree(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      {
        RezCheck z(rez);
        // Pack the users of this node
        size_t num_users = precise_users.size() + imprecise_users.size() + 
                            single_users.size();
        rez.serialize(num_users);
        for (typename std::list<UT>::const_iterator it = precise_users.begin();
              it != precise_users.end(); it++)
        {
          rez.serialize(*it);
        }
        for (typename std::list<UT>::const_iterator it = 
              imprecise_users.begin(); it != imprecise_users.end(); it++)
        {
          rez.serialize(*it);
        }
        for (typename std::list<std::pair<UT,unsigned> >::const_iterator it =
              single_users.begin(); it != single_users.end(); it++)
        {
          rez.serialize(it->first);
        }
        // Then pack what is necessary to make all of the child nodes
        size_t num_children = children.size() + single_children.size(); 
        rez.serialize(num_children);
        for (typename std::set<FieldTree<UT>*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          rez.serialize((*it)->local_mask);
          rez.serialize((*it)->merge_node);
        }
        for (typename std::set<FieldTree<UT>*>::const_iterator it = 
              single_children.begin(); it != single_children.end(); it++)
        {
          rez.serialize((*it)->local_mask);
          rez.serialize((*it)->merge_node);
        }
      }
      // Finally pack each of the children
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        (*it)->pack_field_tree(rez);
      }
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            single_children.begin(); it != single_children.end(); it++)
      {
        (*it)->pack_field_tree(rez);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::unpack_field_tree(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // This maintains the order in which we should unpack the children
      typename std::vector<FieldTree<UT>*> unpacked_children;
      {
        DerezCheck z(derez);
        // Unpack users
        size_t num_users;
        derez.deserialize(num_users);
        for (unsigned idx = 0; idx < num_users; idx++)
        {
          UT user;
          derez.deserialize(user);
          if (FieldMask::pop_count(user.field_mask) == 1)
            add_single_user(user.field_mask.find_first_set(), user);
          else if (user.field_mask == local_mask)
            add_user(user, true/*precise*/);
          else
            add_user(user, false/*precise*/);
        }
        // Unpack and make child nodes
        size_t num_children;
        derez.deserialize(num_children);
        unpacked_children.resize(num_children);
        for (unsigned idx = 0; idx < num_children; idx++)
        {
          FieldMask child_mask;
          derez.deserialize(child_mask);
          bool merge;
          derez.deserialize(merge);
          FieldTree<UT> *child = new FieldTree<UT>(child_mask, merge);
          unpacked_children[idx] = child;
          add_child(child);
        }
      }
      // Then have each child unpack itself
      for (unsigned idx = 0; idx < unpacked_children.size(); idx++)
      {
        unpacked_children[idx]->unpack_field_tree(derez);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    void FieldTree<UT>::analyze_recurse(const FieldMask &mask, AT &analyzer)
    //--------------------------------------------------------------------------
    {
      analyzer.begin_node(this);
      const bool local_dominated = !(local_mask - mask);
      analyze_precise<AT>(analyzer);
      if (local_dominated)
      {
        // Local dominated so no need to do any checks
        analyze_imprecise<AT,false/*checks*/>(mask, analyzer);
        analyze_single<AT,false/*checks*/>(mask, analyzer);
      }
      else
      {
        analyze_imprecise<AT,true/*checks*/>(mask, analyzer);
        analyze_single<AT,true/*checks*/>(mask, analyzer);
      }
      // If we're a single node, then we know that we don't have any children
      if (single_node)
      {
        analyzer.end_node(this);
        return;
      }
      if (local_dominated)
      {
        // Once we are local dominated, then everyone below us is
        // also local dominated
        for (typename std::set<FieldTree<UT>*>::const_iterator it =
              children.begin(); it != children.end(); it++)
        {
          (*it)->analyze_no_checks(analyzer);
        }
        for (typename std::set<FieldTree<UT>*>::const_iterator it =
              single_children.begin(); it != single_children.end(); it++)
        {
          (*it)->analyze_no_checks(analyzer);
        }
      }
      else
      {
        // Now figure out which of our children we need to traverse
        for (typename std::set<FieldTree<UT>*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          // Skip any children with disjoint fields
          if ((*it)->local_mask * mask)
            continue;
          (*it)->analyze_recurse(mask, analyzer);
        }
        for (typename std::set<FieldTree<UT>*>::const_iterator it =
              single_children.begin(); it != single_children.end(); it++)
        {
          if (!mask.is_set((*it)->single_index))
            continue;
          (*it)->analyze_recurse(mask, analyzer);
        }
      }
      analyzer.end_node(this);
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    void FieldTree<UT>::analyze_no_checks(AT &analyzer)
    //--------------------------------------------------------------------------
    {
      analyzer.begin_node(this);
      // We already know we need to compare against everyone so
      // there is no need to do any checks
      analyze_precise<AT>(analyzer);
      for (typename std::list<UT>::iterator it = imprecise_users.begin();
            it != imprecise_users.end(); /*nothing*/)
      {
        if (analyzer.analyze(*it))
          it++;
        else
          it = imprecise_users.erase(it);
      }
      for (typename std::list<std::pair<UT,unsigned> >::iterator it = 
            single_users.begin(); it != single_users.end(); /*nothing*/)
      {
        if (analyzer.analyze(it->first))
          it++;
        else
          it = single_users.erase(it);
      }
      if (single_node)
      {
        analyzer.end_node(this);
        return;
      }
      for (typename std::set<FieldTree<UT>*>::const_iterator it =
            children.begin(); it != children.end(); it++)
      {
        (*it)->analyze_no_checks(analyzer);
      }
      for (typename std::set<FieldTree<UT>*>::const_iterator it =
            single_children.begin(); it != single_children.end(); it++)
      {
        (*it)->analyze_no_checks(analyzer);
      }
      analyzer.end_node(this);
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    void FieldTree<UT>::analyze_single(unsigned index, AT &analyzer)
    //--------------------------------------------------------------------------
    {
      analyzer.begin_node(this);
      // only need to analyze these if we are not a single node
      if (!single_node)
      {
        analyze_single_precise<AT>(index, analyzer);
        analyze_single_imprecise<AT>(index, analyzer);
      }
      analyze_single_single<AT>(index, analyzer);
      // If we're a single node, then we know that we don't have any children
      if (single_node)
      {
        analyzer.end_node(this);
        return;
      }
      // Now figure out which of our children we need to traverse
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        // Skip any children with disjoint fields
        if (!(*it)->local_mask.is_set(index))
          continue;
        (*it)->analyze_single(index, analyzer);
      }
      for (typename std::set<FieldTree<UT>*>::const_iterator it =
            single_children.begin(); it != single_children.end(); it++)
      {
        if ((*it)->single_index != index)
          continue;
        (*it)->analyze_single(index, analyzer);
      }
      analyzer.end_node(this);
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    inline void FieldTree<UT>::analyze_precise(AT &analyzer)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<UT>::iterator it = precise_users.begin();
            it != precise_users.end(); /*nothing*/)
      {
        if (analyzer.analyze(*it))
          it++;
        else
          it = precise_users.erase(it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT, bool CHECK>
    inline void FieldTree<UT>::analyze_imprecise(const FieldMask &mask, 
                                                 AT &analyzer)
    //--------------------------------------------------------------------------
    {
      if (CHECK)
      {
        for (typename std::list<UT>::iterator it = imprecise_users.begin();
              it != imprecise_users.end(); /*nothing*/)
        {
          if (mask * it->field_mask)
          {
            it++;
            continue;
          }
          if (analyzer.analyze(*it))
            it++;
          else
            it = imprecise_users.erase(it);
        }
      }
      else
      {
        for (typename std::list<UT>::iterator it = imprecise_users.begin();
              it != imprecise_users.end(); /*nothing*/)
        {
          if (analyzer.analyze(*it))
            it++;
          else
            it = imprecise_users.erase(it);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT, bool CHECK>
    inline void FieldTree<UT>::analyze_single(const FieldMask &mask, 
                                              AT &analyzer)
    //--------------------------------------------------------------------------
    {
      if (CHECK)
      {
        for (typename std::list<std::pair<UT,unsigned> >::iterator it = 
              single_users.begin(); it != single_users.end(); /*nothing*/)
        {
          if (!mask.is_set(it->second))
          {
            it++;
            continue;
          }
          if (analyzer.analyze(it->first))
            it++;
          else
            it = single_users.erase(it);
        }
      }
      else
      {
        for (typename std::list<std::pair<UT,unsigned> >::iterator it = 
              single_users.begin(); it != single_users.end(); /*nothing*/)
        {
          if (analyzer.analyze(it->first))
            it++;
          else
            it = single_users.erase(it);
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    inline void FieldTree<UT>::analyze_single_precise(unsigned index, 
                                                      AT &analyzer)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<UT>::iterator it = precise_users.begin();
            it != precise_users.end(); /*nothing*/)
      {
        if (!it->field_mask.is_set(index))
        {
          it++;
          continue;
        }
        if (analyzer.analyze(*it))
          it++;
        else
          it = precise_users.erase(it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    inline void FieldTree<UT>::analyze_single_imprecise(unsigned index,
                                                        AT &analyzer)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<UT>::iterator it = imprecise_users.begin();
            it != imprecise_users.end(); /*nothing*/)
      {
        if (!it->field_mask.is_set(index))
        {
          it++;
          continue;
        }
        if (analyzer.analyze(*it))
          it++;
        else
          it = imprecise_users.erase(it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT> template<typename AT>
    inline void FieldTree<UT>::analyze_single_single(unsigned index,
                                                     AT &analyzer)
    //--------------------------------------------------------------------------
    {
      for (typename std::list<std::pair<UT,unsigned> >::iterator it = 
            single_users.begin(); it != single_users.end(); /*nothing*/)
      {
        if (index != it->second)
        {
          it++;
          continue;
        }
        if (analyzer.analyze(it->first))
          it++;
        else
          it = single_users.erase(it);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::insert_recurse(const UT &user)
    //--------------------------------------------------------------------------
    {
      if (user.field_mask == local_mask)
      {
        // Base case: if we've arrived at a node with our field mask
        // then we are done
        add_user(user, true/*precise*/);
      }
      else
      {
        // Otherwise compute two sets:
        //  - Overlapping sets which puts an upper bound on the
        //      children that we may traverse
        //  - Dominators which places a lower bound on the 
        //      children that we may traverse
        typename std::map<FieldTree<UT>*,bool/*single*/> overlaps;
        typename std::set<FieldTree<UT>*> dominators;
        for (typename std::set<FieldTree*>::const_iterator it = 
              children.begin(); it != children.end(); it++)
        {
          if ((*it)->local_mask * user.field_mask)
            continue;
          overlaps[*it] = false;
          if (!(user.field_mask - (*it)->local_mask))
            dominators.insert(*it);
        }
        for (typename std::set<FieldTree<UT>*>::const_iterator it =
              single_children.begin(); it != single_children.end(); it++)
        {
          if (!user.field_mask.is_set((*it)->single_index))
            continue;
          overlaps[*it] = true;
          FieldMask copy = user.field_mask;
          copy.unset_bit((*it)->single_index);
          if (!copy)
            dominators.insert(*it);
        }
        // There are three scenarios here:
        //  - No overlaps: make a new node and add it to the children
        //  - One overlap: figure out who dominates who or not at all
        //                  and then decide what to do
        //  - Many overlaps: complicated, see more comments below
        if (overlaps.empty())
        {
          FieldTree<UT> *child_node = new FieldTree<UT>(user.field_mask);
          child_node->add_user(user, true/*precise*/);
          // Add the child to this node
          add_child(child_node);
        }
        else if (overlaps.size() == 1)
        {
          // Three scenarios here:
          // - Current child dominates the new child: if so then just
          //    continue inserting
          // - New child dominates the old child: make a node for the
          //    new child and make the old child a child of the new
          //    child node
          // - Neither dominates: make a new node for the child
          //    and add it to this node
          typename std::map<FieldTree<UT>*,bool>::const_iterator next = 
            overlaps.begin();
          if (!dominators.empty())
          {
            // Old node dominates the new user, continue the traversal
            next->first->insert(user);
          }
          else if (!(next->first->local_mask - user.field_mask))
          {
            // New user dominates the old user
            FieldTree<UT> *child_node = new FieldTree<UT>(user.field_mask);
            child_node->add_user(user, true/*precise*/);
            child_node->add_child(next->first);
            // Remove the old child and add it to the new child
            if (next->second)
              single_children.erase(next->first);
            else
              children.erase(next->first);
            // Now add the new child to this node
            add_child(child_node);
          }
          else
          {
            // Neither one dominates, so make a new node and add it
            FieldTree<UT> *child_node = new FieldTree<UT>(user.field_mask);
            child_node->add_user(user, true/*precise*/);
            add_child(child_node);
          }
        }
        else
        {
          // We had many overlaps.  There are now three possibilities
          // depending on how many dominators we had.
          // No dominators: make a new node for this child and add it.
          //    In the process pull in any current children that the
          //    new node now dominates.
          // One dominator: continue down the dominated node
          // Many dominators: make a new node to represent the union
          //    of all the dominators and then pull all the dominators
          //    inside of it.  Adding the user to this node will 
          //    likely be imprecise.
          if (dominators.empty())
          {
            FieldTree<UT> *child_node = new FieldTree<UT>(user.field_mask);
            child_node->add_user(user, true/*precise*/);
            // Pull in any other children that the new node dominates.
            for (typename std::map<FieldTree<UT>*,bool>::const_iterator
                  it = overlaps.begin(); it != overlaps.end(); it++)
            {
              // Skip anything we don't dominate
              if (!!(it->first->local_mask - user.field_mask))
                continue;
              if (it->second)
                single_children.erase(it->first);
              else
                children.erase(it->first);
              child_node->add_child(it->first);
            }
            add_child(child_node);
          }
          else if (dominators.size() == 1)
          {
            FieldTree<UT> *dominator = *(dominators.begin());
            dominator->insert(user);
          }
          else
          {
            // No point in making a new merge node if we
            // already are a merge node.
            if (merge_node)
            {
              add_user(user, false/*precise*/);
            }
            else
            {
              // Make a node that is the union of all the dominators and
              // place the user in it.  Then move all the dominators
              // inside of that node.
              FieldMask dominator_mask;
              for (typename std::set<FieldTree<UT>*>::const_iterator it =
                    dominators.begin(); it != dominators.end(); it++)
              {
                dominator_mask |= (*it)->local_mask;
              }
              // If the union of the dominator mask is this node, then
              // just record ourself here imprecisely since we already
              // know we are not equal to the local mask.
              if (dominator_mask == local_mask)
              {
                add_user(user, false/*precise*/);
              }
              else
              {
                FieldTree<UT> *dominator_node = merge_dominators(dominators,
                                                            dominator_mask);
                dominator_node->add_user(user, 
                    (dominator_mask==user.field_mask));
                // Finally add the dominator node to this node
                add_child(dominator_node);
              }
            }
          }
        }
      }
    }  

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::insert_single(unsigned index, const UT &user)
    //--------------------------------------------------------------------------
    {
      // Handle the base case
      if (single_node)
      {
        // Only here if the indexes match
        add_single_user(index, user);
        return;
      }
      // Now check the single users to see if we find the node
      // we're looking for.  If we do then we're done.
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            single_children.begin(); it != single_children.end(); it++)
      {
        if ((*it)->single_index == index)
        {
          (*it)->insert_single(index, user);
          return;
        }
      }
      // Now find overlaps/dominators.  Since we are a single field
      // mask we know that any overlap by definition dominates us.
      typename std::set<FieldTree<UT>*> dominators;
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        if ((*it)->local_mask.is_set(index))
          dominators.insert(*it);
      }
      // See how many dominators we had
      if (dominators.empty())
      {
        // Make a new single node and add it
        FieldTree<UT> *child = new FieldTree<UT>(user.field_mask);
        child->add_single_user(index, user);
        // Add the child to this node
        add_child(child);
      }
      else if (dominators.size() == 1)
      {
        // Continue the traversal
        FieldTree<UT> *dominator = *(dominators.begin());
        dominator->insert_single(index, user);
      }
      else
      {
        // No point in making a new merge node if
        // we already are a merge node.
        if (merge_node)
        {
          add_single_user(index, user);
        }
        else
        {
          // Multiple dominators
          FieldMask dominator_mask;
          for (typename std::set<FieldTree<UT>*>::const_iterator it =
                dominators.begin(); it != dominators.end(); it++)
          {
            dominator_mask |= (*it)->local_mask;
          }
          if (dominator_mask == local_mask)
          {
            add_single_user(index, user);
          }
          else
          {
            FieldTree<UT> *dominator = merge_dominators(dominators, 
                                                    dominator_mask);
            dominator->add_single_user(index, user);
            add_child(dominator);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    inline FieldTree<UT>* FieldTree<UT>::merge_dominators(
        const std::set<FieldTree<UT>*> &dominators, const FieldMask &dom_mask)
    //--------------------------------------------------------------------------
    {
      FieldTree<UT> *dominator_node = new FieldTree<UT>(dom_mask, true/*dom*/);
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            dominators.begin(); it != dominators.end(); it++)
      {
        // We know that all dominators have more then one field
        // because if there were multiple dominators and one of
        // them had only one field, then the base user mask would
        // only be one field.  This means that the dominator with
        // only one field would be dominated by all the other 
        // dominators which violates the invariant that no child
        // dominates any others which is always maintained.
        children.erase(*it);
        // If the dominator is precise then add it, otherwise flatten
        // it into this node.
        if (!(*it)->merge_node)
          dominator_node->add_child(*it);
        else
        {
          (*it)->subsume_node(dominator_node);
          // Now delete the node
          delete (*it);
        }
      }
#ifdef DEBUG_HIGH_LEVEL
      dominator_node->check_state();
#endif
      return dominator_node;
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    inline void FieldTree<UT>::subsume_node(FieldTree<UT> *dominator_node)
    //--------------------------------------------------------------------------
    {
      // First copy up the users
      for (typename std::list<UT>::const_iterator it = precise_users.begin();
            it != precise_users.end(); it++)
      {
        dominator_node->add_user(*it, false/*precise*/);
      }
      precise_users.clear();
      for (typename std::list<UT>::const_iterator it = imprecise_users.begin();
            it != imprecise_users.end(); it++)
      {
        dominator_node->add_user(*it, false/*precise*/);
      }
      imprecise_users.clear();
      for (typename std::list<std::pair<UT,unsigned> >::const_iterator it = 
            single_users.begin(); it != single_users.end(); it++)
      {
        dominator_node->add_single_user(it->second, it->first);
      }
      single_users.clear();
      // Copy up the single children
      for (typename std::set<FieldTree<UT>*>::const_iterator it =
            single_children.begin(); it != single_children.end(); it++)
      {
        dominator_node->add_child(*it);
      }
      // Clear them so we don't delete them when we delete the node
      single_children.clear();
      for (typename std::set<FieldTree<UT>*>::const_iterator it = 
            children.begin(); it != children.end(); it++)
      {
        dominator_node->add_child(*it);
      }
      // clear them so we don't delete them when we delete the node
      children.clear();
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    inline void FieldTree<UT>::add_user(const UT &user, bool precise)
    //--------------------------------------------------------------------------
    {
      if (precise)
        precise_users.push_back(user);
      else
        imprecise_users.push_back(user);
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    inline void FieldTree<UT>::add_single_user(unsigned index, const UT &user)
    //--------------------------------------------------------------------------
    {
      single_users.push_back(std::pair<UT,unsigned>(user,index));
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    inline void FieldTree<UT>::add_child(FieldTree<UT> *child)
    //--------------------------------------------------------------------------
    {
      if (child->single_node)
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(FieldMask::pop_count(child->local_mask) == 1);
        assert(child->local_mask != local_mask);
        assert(!(child->local_mask - local_mask));
#endif
        single_children.insert(child);
      }
      else
      {
#ifdef DEBUG_HIGH_LEVEL
        assert(FieldMask::pop_count(child->local_mask) > 1);
        assert(child->local_mask != local_mask);
        assert(!(child->local_mask - local_mask));
#endif
        children.insert(child);
      }
    }

    //--------------------------------------------------------------------------
    template<typename UT>
    void FieldTree<UT>::check_state(void)
    //--------------------------------------------------------------------------
    {
      // Check to make sure that all the children are non-overlapping
      for (typename std::set<FieldTree<UT>*>::const_iterator it1 = 
            children.begin(); it1 != children.end(); it1++)
      {
        for (typename std::set<FieldTree<UT>*>::const_iterator it2 = 
              children.begin(); it2 != it1; it2++)
        {
          assert(!!((*it1)->local_mask - (*it2)->local_mask));
          assert(!!((*it2)->local_mask - (*it1)->local_mask));
        }
      }
      for (typename std::set<FieldTree<UT>*>::const_iterator sit = 
            single_children.begin(); sit != single_children.end(); sit++)
      {
        for (typename std::set<FieldTree<UT>*>::const_iterator cit = 
              children.begin(); cit != children.end(); cit++)
        {
          assert(!((*cit)->local_mask.is_set((*sit)->single_index)));
        }
        for (typename std::set<FieldTree<UT>*>::const_iterator cit = 
              single_children.begin(); cit != sit; cit++)
        {
          assert((*sit)->single_index != (*cit)->single_index);
        }
      }
    }
    
  };
};

#endif // __LEGION_FIELD_TREE_H__

// EOF

