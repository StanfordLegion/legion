/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <algorithm>
#include <typeinfo>
#include "mappers/wrapper_mapper.h"
#include <vector>

#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_BREADTH_FIRST          false
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8


namespace Legion {
  namespace Mapping{

    //Static Variables. 
    /*procs_map, procs_map_init, print_tasks and stop_tasks store information 
      about the processors and tasks being monitored.
      The owner processor broadcasts the information to all the processors 
      and local owner of each node then stores it in their static variables.
      Only the owner processor communicates with the user.*/
    std::set<Memory> WrapperMapper::all_mems;
    std::set<Processor> WrapperMapper::all_procs;
    std::map<Processor, int> WrapperMapper::procs_map;
    std::map<int, int> WrapperMapper::procs_map_int;
    std::map<int, int> WrapperMapper::methods_map;
    //std::map<std::string, int> WrapperMapper::tasks_map;
    std::vector<std::string> WrapperMapper::print_tasks;
    std::vector<std::string> WrapperMapper::stop_tasks;
    bool WrapperMapper::inputtaken=0;
    bool WrapperMapper::databroadcasted = 0;
    Processor WrapperMapper::ownerprocessor;
    Processor WrapperMapper::localowner;
    MapperEvent WrapperMapper::mapevent;			
    int WrapperMapper::broadcastcount=0;	

    WrapperMapper::WrapperMapper(Mapper* dmapper,MapperRuntime *rt, 
	Machine machine, Processor local):Mapper(rt), dmapper(dmapper), mrt(rt),
    local_proc(local), local_kind(local.kind()), 
    node_id(local.address_space()), machine(machine),
    max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
    max_steal_count(STATIC_MAX_STEAL_COUNT),
    breadth_first_traversal(STATIC_BREADTH_FIRST),
    stealing_enabled(STATIC_STEALING_ENABLED),
    max_schedule_count(STATIC_MAX_SCHEDULE_COUNT){
      machine.get_all_processors(all_procs);
      machine.get_all_memories(all_mems);
      if (!inputtaken && node_id==0){
	get_input(); //First proc of node 0 gets the input from the user
	inputtaken=1;
	ownerprocessor = local; //First proc of node 0 is the owner proc
	localowner = local;	
	/*Since only select_task_options() is wrapped, 
	  there is no need to ask the user to add methods. 
	  Hence, select_task_options() is added by default.*/
	methods_map.insert(std::pair<int, int>(1,0));	
      }
      else if (!inputtaken){
	inputtaken =1;
	localowner = local;     
	methods_map.insert(std::pair<int, int>(1,0));	
      }
    }
    WrapperMapper::~WrapperMapper(){
      //Debugging
      /*std::cout<<local_proc.id<<"-> Owner:"<<ownerprocessor.id<<"\n";
	std::cout<<local_proc.id<<"-> The tasks added are: ";
      //for (std::map<std::string, int>::const_iterator i = tasks_map.begin();
      i != tasks_map.end(); ++i) std::cout<< i->first << "  ";
      for (std::vector<std::string>::const_iterator i = print_tasks.begin();
      i != print_tasks.end(); ++i) std::cout<< *i << "  ";
      std::cout<<"\n";
      std::cout<<local_proc.id<<"-> The processors added are: ";
      for (std::map<Processor,int>::const_iterator it = procs_map.begin(); 
      it != procs_map.end(); ++it) std::cout<< it->first.id << "   ";
      std::cout<<"\n";*/

    }

    //Helper functions	
    bool is_number(const std::string& s)
    {
      std::string::const_iterator it = s.begin();
      while (it != s.end() && std::isdigit(*it)) ++it;
      return !s.empty() && it == s.end();
    }

    bool WrapperMapper::InputNumberCheck(std::string strUserInput)
    {
      for (unsigned int nIndex=0; nIndex < strUserInput.length(); nIndex++)
      {
	if (!std::isdigit(strUserInput[nIndex])) return false;
      }
      return 1;
    }		

    template <typename T>
      std::string NumberToString ( T Number )
      {
	std::stringstream ss;
	ss << Number;
	return ss.str();
      }

    /*Serialize the data in tasks_map and procs_map and convert it into a 
      string so that it can be sent to all the processors*/
    std::string WrapperMapper::Serialize
      (const std::map<std::string, int> &tasks_map, 
       const std::map<int, int> &procs_map ){
	std::string send_string, temp;

	for (std::map<int, int>::const_iterator i = 
	    procs_map.begin(); i!=procs_map.end(); ++i){
	  send_string = send_string + NumberToString(i->first) + 
	    NumberToString(i->second) + "\\";
	}
	send_string = send_string + "#";

	for (std::map<std::string, int>::const_iterator i = 
	    tasks_map.begin(); i != tasks_map.end(); ++i){
	  send_string = send_string + i->first + 
	    NumberToString(i->second) + "\\";
	}

	return send_string;
      }

    std::string WrapperMapper::Serialize
      (const std::vector<std::string> &tasks_print,
       const std::vector<std::string> &tasks_stop, 
       const std::map<int, int> &procs_map){

	std::string send_string;
	for (std::map<int, int>::const_iterator i =
	    procs_map.begin(); i!=procs_map.end(); ++i){
	  send_string = send_string + NumberToString(i->first) +
	    NumberToString(i->second) + "\\";
	}
	send_string = send_string + "#";

	for (std::vector<std::string>::const_iterator i =
	    tasks_print.begin(); i != tasks_print.end(); ++i){
	  send_string = send_string + *i +  "\\";
	}
	send_string = send_string + "#";

	for (std::vector<std::string>::const_iterator i =
	    tasks_stop.begin(); i != tasks_stop.end(); ++i){
	  send_string = send_string + *i + "\\";
	}

	return send_string;
      } 

    //Deserialize the received string and store the data in the maps
    void WrapperMapper::Deserialize(std::string rec_string){
      std::size_t hash_pos  = rec_string.find("#");
      std::string  procs_str = rec_string.substr(0, hash_pos);

      std::string tasks_str =
	rec_string.substr(hash_pos+1, rec_string.size() - hash_pos);
      hash_pos = tasks_str.find("#");

      std::string  print_tasks_str = tasks_str.substr(0, hash_pos);
      std::string stop_tasks_str =
	tasks_str.substr(hash_pos+1, rec_string.size() - hash_pos);

      print_tasks.clear();
      stop_tasks.clear();

      std::string delim = "\\";
      std::map<std::string, int> map_tasks;			
      std::string token;
      std::size_t pos = 0;

      while ((pos = print_tasks_str.find(delim)) != std::string::npos){
	token = print_tasks_str.substr(0, pos);
	print_tasks.push_back(token);
	print_tasks_str.erase(0, pos + delim.length());
      }

      while ((pos = stop_tasks_str.find(delim)) != std::string::npos){
	token = stop_tasks_str.substr(0, pos);
	stop_tasks.push_back(token);
	stop_tasks_str.erase(0, pos + delim.length());
      }

      int ip;
      std::set<Processor>::iterator it;
      std::vector<Processor> procs_print;
      std::vector<Processor> procs_stop;
      while ((pos = procs_str.find(delim)) != std::string::npos){
	token = procs_str.substr(0, pos);
	ip = std::atoi(token.substr(0, token.size()-1).c_str());
	if ((unsigned)ip<all_procs.size()){
	  it = all_procs.begin();
	  std::advance(it, ip);
	  procs_map.insert(
	      std::pair<Processor,int>(*it, (int)(token.at(token.size()-1))));				
	}
	procs_str.erase(0, pos + delim.length());
      }

      std::set<Processor>::iterator ito;
      ito = all_procs.begin();
      std::advance(ito, 1);
      ownerprocessor = *ito;

    }

    //Get the input from the user
    void WrapperMapper::get_input(const MapperContext(ctx)){
      std::string strValue;
      std::map<int, std::string> function_map;
      int Value, pValue;

      function_map[1] = "select_task_options"; 
      function_map[2] = "select_tasks_to_schedule";
      function_map[3] = "target_task_steal"; 
      function_map[4] = "permit_task_steal";
      function_map[5] = "slice_domain"; 
      function_map[6] = "pre_map_task";
      function_map[7] = "select_task_variant"; 
      function_map[8] = "map_task";
      function_map[9] = "post_map_task"; 
      function_map[10] = "map_copy";
      function_map[11] = "map_inline"; 
      function_map[12] = "map_must_epoch";
      function_map[13] = "notify_mapping_result"; 
      function_map[14] = "notify_mapping_failed";
      function_map[15] = "rank_copy_targets"; 
      function_map[16] = "rank_copy_sources";
      function_map[17] = "Other";

      std::cout<< "Enter \"help\" to see the list of commands.";
      std::cout<<" Enter \"exit\" to exit.\n";
      std::cout<<">    ";
      while (1)
      {
	getline(std::cin, strValue); 
	std::string nameValue;
	std::string intValue;

	//Add a task for which the information needs to be printed
	if (strValue.compare(0,12,"print task +")==0){
	  nameValue=strValue.substr(12);

	  std::vector<std::string>::iterator it = 
	    std::find(print_tasks.begin(), print_tasks.end(), nameValue);  
	  std::vector<std::string>::iterator its = 
	    std::find(stop_tasks.begin(), stop_tasks.end(), nameValue);  
	  if (it==print_tasks.end())
	  {
	    pValue=2;
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i =
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	  else{
	    print_tasks.erase(it);
	    if (its != stop_tasks.end()) stop_tasks.erase(its);
	    print_tasks.push_back(nameValue);
	    pValue=2;
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	}

	//Add a task for which program execution needs to stop
	else if (strValue.compare(0,11,"stop task +")==0){
	  nameValue=strValue.substr(11);

	  std::vector<std::string>::iterator itp = 
	    std::find(print_tasks.begin(), print_tasks.end(), nameValue);  
	  std::vector<std::string>::iterator its = 
	    std::find(stop_tasks.begin(), stop_tasks.end(), nameValue);  

	  if (itp==print_tasks.end())
	  {
	    pValue=1;
	    stop_tasks.push_back(nameValue);
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i =
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	  else{
	    print_tasks.erase(itp);
	    stop_tasks.erase(its);
	    pValue=1;
	    stop_tasks.push_back(nameValue);
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	}

	//Add a method/function for which the information needs to be printed
	// (not needed at the moment)
	else if (strValue.compare(0,14,"print method +")==0){
	  intValue=strValue.substr(14);
	  if(InputNumberCheck(intValue)){
	    Value = std::atoi(intValue.c_str());
	    if (Value>0 && Value<18){ 
	      std::map<int, int>::iterator it = methods_map.find(Value);
	      if (it==methods_map.end()){
		pValue=2;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i = 
		    methods_map.begin(); i != methods_map.end(); ++i) 
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	      else{
		methods_map.erase(it);
		pValue=2;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i = 
		    methods_map.begin(); i != methods_map.end(); ++i)
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	    }
	    else std::cout<<"Method number should be between 1 and 17\n>    ";
	  }
	  else std::cout<<"Method ID not a number\n>    ";
	}

	//Add a method/function for which program execution needs to stop 
	//(not needed at the moment)
	else if (strValue.compare(0,13,"stop method +")==0){
	  intValue=strValue.substr(13);
	  if(InputNumberCheck(intValue)){
	    Value = std::atoi(intValue.c_str());
	    if (Value>0 && Value<18){ 
	      std::map<int, int>::iterator it = methods_map.find(Value);
	      if (it==methods_map.end()){
		pValue=1;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i = 
		    methods_map.begin(); i != methods_map.end(); ++i) 
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	      else{
		methods_map.erase(it);
		pValue=1;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i =
		    methods_map.begin(); i != methods_map.end(); ++i)
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	    }
	    else std::cout<<"Method number should be between 1 and 17\n>    ";
	  }
	  else std::cout<<"Method ID not a number\n>    ";
	}

	//Add a processor for which the information needs to be printed
	else if (strValue.compare(0,17,"print processor +")==0){
	  intValue=strValue.substr(17);
	  std::set<Processor>::iterator it;
	  if (is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if ((unsigned)i<all_procs.size()){
	      it = all_procs.begin();
	      std::advance(it, i);
	      std::map<Processor, int>::iterator ite= procs_map.find(*it);
	      if (ite!=procs_map.end() ) procs_map.erase(ite);				
	      pValue=2;
	      procs_map.insert(std::pair<Processor,int>(*it,pValue));
	      procs_map_int.insert(std::pair<int, int>(i, pValue));
	      std::cout<<"The processors added are: ";
	      for (std::map<Processor,int>::const_iterator it = 
		  procs_map.begin(); it != procs_map.end(); ++it) 
		std::cout<< it->first.id << "   ";
	      std::cout<<"\n>    ";
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";			
	}

	//Add a processor for which program execution needs to stop
	else if (strValue.compare(0,16,"stop processor +")==0){
	  intValue=strValue.substr(16);
	  std::set<Processor>::iterator it;
	  if (is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if ((unsigned)i<all_procs.size()){
	      it = all_procs.begin();
	      std::advance(it, i);
	      std::map<Processor, int>::iterator ite= procs_map.find(*it);
	      if (ite!=procs_map.end()) procs_map.erase(ite);				
	      pValue=1;
	      procs_map.insert(std::pair<Processor,int>(*it,pValue));
	      procs_map_int.insert(std::pair<int, int>(i, pValue));
	      std::cout<<"The processors added are: ";
	      for (std::map<Processor,int>::const_iterator it = 
		  procs_map.begin(); it != procs_map.end(); ++it)
		std::cout<< it->first.id << "   ";
	      std::cout<<"\n>    ";
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";			
	}

	/*else if (strValue.compare(0,14,"print memory +")==0){
	  intValue=strValue.substr(14);
	  std::set<Memory>::iterator it;
	  if (is_number(intValue)){
	  int i=std::atoi(intValue.c_str())-1;
	  if((unsigned)i<all_mems.size()){
	  it = all_mems.begin();
	  std::advance(it,i);
	  std::map<Memory, int>::iterator itm = mems_map.find(*it);
	  if (itm!=mems_map.end()) mems_map.erase(itm);
	  pValue=1;									
	  mems_map.insert(std::pair<Memory,int>(*it,pValue));

	  std::cout<<"The memories added are: ";
	  for (std::map<Memory,int>::const_iterator it = 
	  mems_map.begin(); it != mems_map.end(); ++it) 
	  std::cout<<it->first.id<<"		";
	  std::cout<<"\n>    ";
	  }
	  else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";
	  }

	  else if (strValue.compare(0,13,"stop memory +")==0){
	  intValue=strValue.substr(13);
	  std::set<Memory>::iterator it;
	  if (is_number(intValue)){
	  int i=std::atoi(intValue.c_str())-1;
	  if((unsigned)i<all_mems.size()){
	  it = all_mems.begin();
	  std::advance(it,i);
	  std::map<Memory, int>::iterator itm = mems_map.find(*it);
	  if (itm!=mems_map.end()) mems_map.erase(itm);
	  pValue=0;									
	  mems_map.insert(std::pair<Memory,int>(*it,pValue));

	  std::cout<<"The memories added are: ";
	  for (std::map<Memory,int>::const_iterator it = 
	  mems_map.begin(); it != mems_map.end(); ++it)
	  std::cout<<it->first.id<<"		";
	  std::cout<<"\n>    ";

	  }
	  else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";
	  }*/

	//Remove a task from the tasks map
	  else if (strValue.compare(0,6,"task -")==0){
	    nameValue=strValue.substr(6);

	    std::vector<std::string>::iterator itp = 
	      std::find(print_tasks.begin(), print_tasks.end(),nameValue);  
	    std::vector<std::string>::iterator its = 
	      std::find(stop_tasks.begin(), stop_tasks.end(),nameValue);  

	    if (itp!=print_tasks.end())
	    {
	      print_tasks.erase(itp);
	      stop_tasks.erase(its);
	      std::cout<<"The tasks added are: ";
	      for (std::vector<std::string>::const_iterator i = 
		  print_tasks.begin(); i != print_tasks.end(); ++i) 
		std::cout<< *i << "  ";
	      std::cout<<"\n>    ";
	    }
	    else{
	      std::cout<<"Task "<<nameValue<<" not present\n";
	      std::cout<<"The tasks added are: ";
	      for (std::vector<std::string>::const_iterator i = 
		  print_tasks.begin(); i != print_tasks.end(); ++i) 
		std::cout<< *i << "  ";
	      std::cout<<"\n>    ";
	    }
	  }


	  //Remove a method/function from the methods map
	  else if (strValue.compare(0,8,"method -")==0){
	    intValue=strValue.substr(8);
	    if(InputNumberCheck(intValue)){
	      Value = std::atoi(intValue.c_str());
	      if (Value>0 && Value<18){
		std::map<int, int>::iterator it = methods_map.find(Value);
		if (it!=methods_map.end()){
		  methods_map.erase(it);				
		  std::cout<<"The methods added are: ";
		  for (std::map<int, int>::const_iterator i = 
		      methods_map.begin(); i != methods_map.end(); ++i) 
		    std::cout<< function_map[i->first] << "  ";
		  std::cout<<"\n>    ";
		}
		else std::cout<<"Method not present.\n>    "; 
	      }
	      else std::cout<<"Method number should be between 1 and 17\n>    ";
	    }
	    else std::cout<<"Method ID not a number\n>    ";
	  }

	  //Remove a processor from the processors map
	  else if (strValue.compare(0,11,"processor -")==0){
	    intValue=strValue.substr(11);
	    std::set<Processor>::iterator it;
	    std::map<Processor, int>::iterator ite;
	    if (is_number(intValue)){
	      int i=std::atoi(intValue.c_str())-1;
	      if ((unsigned)i<all_procs.size()){
		it = all_procs.begin();
		std::advance(it, i);
		std::map<Processor, int>::iterator ite= procs_map.find(*it);
		std::map<int, int>::iterator ite_int = procs_map_int.find(i);
		if (ite!=procs_map.end() ){
		  procs_map.erase(ite);
		  procs_map_int.erase(ite_int);				
		  std::cout<<"The processors added are: ";
		  for (std::map<Processor,int>::const_iterator it = 
		      procs_map.begin(); it != procs_map.end(); ++it) 
		    std::cout<< it->first.id << "   ";
		  std::cout<<"\n>    ";
		}
		else{
		  std::cout<<"Processor not present.\n";
		  std::cout<<"The processors added are: ";
		  for (std::map<Processor,int>::const_iterator it = 
		      procs_map.begin(); it != procs_map.end(); ++it) 
		    std::cout<< it->first.id << "   ";
		  std::cout<<"\n>    ";
		}
	      }
	      else std::cout<<"Invalid number entered\n>    ";
	    }
	    else std::cout<<"Invalid input\n>    ";

	  }

	  /*else if (strValue.compare(0,8,"memory -")==0){
	    intValue=strValue.substr(8);
	    std::set<Memory>::iterator it;
	    std::map<Memory, int>::iterator ite;
	    if(is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if((unsigned)i<all_mems.size()){
	    it = all_mems.begin();
	    std::advance(it, i);
	    std::map<Memory, int>::iterator ite=mems_map.find(*it);
	    if(ite!=mems_map.end()){
	    mems_map.erase(ite);
	    std::cout<<"The memories added are:	";
	    for (std::map<Memory,int>::const_iterator it = 
	    mems_map.begin(); it!=mems_map.end(); ++it) 
	    std::cout<<it->first.id<<"		";
	    std::cout<<"\n>    ";
	    }
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	    }
	    else std::cout<<"Invalid input\n>    ";


	    }*/

	  //Help
	    else if (strValue.compare("help")==0){
	      std::cout<<"Following are the commands that can be executed:\n";
	      std::cout<<"task +<task_id> --> To add a task to be monitored \n";
	      std::cout<<"task -<task_id> --> To remove a task from the lists";
	      std::cout<<" of tasks which are being monitored \n";
	      std::cout<<"methods --> To see the list of methods with their ";
	      std::cout<<"corresponding ids\n";
	      std::cout<<"method +<method_id> --> To add a method to be i";
	      std::cout<<"monitored\n";
	      std::cout<<"method -<method_id> --> To remove a method from the ";
	      std::cout<<"lists of methods which are being monitored \n";
	      std::cout<<"processors --> To see the list of processor with ";
	      std::cout<<"their corresponding ids\n";
	      std::cout<<"processor +<processor_id> --> To add a processor to ";
	      std::cout<<"be monitored\n";
	      std::cout<<"processor -<processor_id> --> To remove a processor ";
	      std::cout<<"from the lists of processors which are being ";
	      std::cout<<"monitored \n";
	      std::cout<<">    ";
	    }

	    //List all the methods
	    else if (strValue.compare("methods")==0){
	      for(std::map<int, std::string >::const_iterator it = 
		  function_map.begin(); it != function_map.end(); ++it)
	      {
		std::cout << it->first << ". " << it->second << " " << "\n";
	      }
	      std::cout<<">    ";
	    }
	    //List all the processors
	    else if (strValue.compare("processors")==0){
	      int i=0;
	      std::set<Processor>::iterator it;
	      for ( it = all_procs.begin();
		  it != all_procs.end(); it++)
	      {
		i++;
		Processor::Kind k = it->kind();
		if (k == Processor::UTIL_PROC) 
		  std::cout<<i<<". Utility Processor ID:"<<it->id<<"\n";
		else 
		  std::cout<<i<<". Processor ID: "<<it->id<<"  Kind:"<<k<<"\n";
	      }
	      std::cout<<">    ";
	    }
	    /*
	       else if (strValue.compare("memories")==0){
	       int i=0;
	       std::set<Memory>::iterator it;
	       for ( it = all_mems.begin();
	       it != all_mems.end(); it++)
	       {
	       i++;
	       std::cout<<i<<". Memory ID: "<<it->id<<"  Capacity: ";
	       std::cout<<it->capacity()<<"  Kind: "<<it->kind()<<"\n";
	       }
	       std::cout<<">    ";
	       }
	     */
	    //Exit 
	    else if (strValue.compare("exit")==0){
	      std::string send_message = 
		Serialize(print_tasks,stop_tasks, procs_map_int);

	      int send_size = send_message.size()+1;
	      char send_mess_chars[send_size];
	      strcpy(send_mess_chars, send_message.c_str());
	      void *message_point = &send_mess_chars;
	      mrt->broadcast(ctx, message_point, send_size*sizeof(char));  
	      //Broadcast the information to all processors on exit            	
	      break;
	    }

	    else std::cout<<"Invalid Command\n>    ";

      }
    }

    /*Overloaded version of the previous function to get inputs at the start and
      without broadcast on exit*/
    void WrapperMapper::get_input(){
      std::string strValue;
      std::map<int, std::string> function_map;
      int Value, pValue;

      function_map[1] = "select_task_options"; 
      function_map[2] = "select_tasks_to_schedule";
      function_map[3] = "target_task_steal"; 
      function_map[4] = "permit_task_steal";
      function_map[5] = "slice_domain"; 
      function_map[6] = "pre_map_task";
      function_map[7] = "select_task_variant"; 
      function_map[8] = "map_task";
      function_map[9] = "post_map_task"; 
      function_map[10] = "map_copy";
      function_map[11] = "map_inline"; 
      function_map[12] = "map_must_epoch";
      function_map[13] = "notify_mapping_result"; 
      function_map[14] = "notify_mapping_failed";
      function_map[15] = "rank_copy_targets"; 
      function_map[16] = "rank_copy_sources";
      function_map[17] = "Other";

      std::cout<< "Enter \"help\" to see the list of commands. ";
      std::cout<<" Enter \"exit\" to exit.\n";
      std::cout<<">    ";
      while (1)
      {
	getline(std::cin, strValue); 
	std::string nameValue;
	std::string intValue;
	if (strValue.compare(0,12,"print task +")==0){
	  nameValue=strValue.substr(12);
	  std::vector<std::string>::iterator it = 
	    std::find(print_tasks.begin(), print_tasks.end(), nameValue);  
	  std::vector<std::string>::iterator its = 
	    std::find(stop_tasks.begin(), stop_tasks.end(), nameValue);  
	  if (it==print_tasks.end())
	  {
	    pValue=2;
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	  else{
	    print_tasks.erase(it);
	    if (its != stop_tasks.end()) stop_tasks.erase(its);
	    print_tasks.push_back(nameValue);
	    pValue=2;
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	}

	//Add a task for which program execution needs to stop
	else if (strValue.compare(0,11,"stop task +")==0){
	  nameValue=strValue.substr(11);

	  std::vector<std::string>::iterator itp = 
	    std::find(print_tasks.begin(), print_tasks.end(), nameValue);  
	  std::vector<std::string>::iterator its = 
	    std::find(stop_tasks.begin(), stop_tasks.end(), nameValue);  
	  if (itp==print_tasks.end())
	  {
	    pValue=1;
	    stop_tasks.push_back(nameValue);
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	  else{
	    print_tasks.erase(itp);
	    stop_tasks.erase(its);
	    pValue=1;
	    stop_tasks.push_back(nameValue);
	    print_tasks.push_back(nameValue);
	    std::cout<<"The tasks added are: ";
	    for (std::vector<std::string>::const_iterator i = 
		print_tasks.begin(); i != print_tasks.end(); ++i) 
	      std::cout<< *i << "  ";
	    std::cout<<"\n>    ";
	  }
	}

	else if (strValue.compare(0,14,"print method +")==0){
	  intValue=strValue.substr(14);
	  if(InputNumberCheck(intValue)){
	    Value = std::atoi(intValue.c_str());
	    if (Value>0 && Value<18){ 
	      std::map<int, int>::iterator it = methods_map.find(Value);
	      if (it==methods_map.end()){
		pValue=2;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i = 
		    methods_map.begin(); i != methods_map.end(); ++i) 
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	      else{
		methods_map.erase(it);
		pValue=2;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i = 
		    methods_map.begin(); i != methods_map.end(); ++i)
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	    }
	    else std::cout<<"Method number should be between 1 and 17\n>    ";
	  }
	  else std::cout<<"Method ID not a number\n>    ";
	}

	else if (strValue.compare(0,13,"stop method +")==0){
	  intValue=strValue.substr(13);
	  if(InputNumberCheck(intValue)){
	    Value = std::atoi(intValue.c_str());
	    if (Value>0 && Value<18){ 
	      std::map<int, int>::iterator it = methods_map.find(Value);
	      if (it==methods_map.end()){
		pValue=1;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i =
		    methods_map.begin(); i != methods_map.end(); ++i)
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	      else{
		methods_map.erase(it);
		pValue=1;
		methods_map.insert(std::pair<int, int>(Value,pValue));
		std::cout<<"The methods added are: ";
		for (std::map<int, int>::const_iterator i =
		    methods_map.begin(); i != methods_map.end(); ++i)
		  std::cout<< function_map[i->first] << "  ";
		std::cout<<"\n>    ";
	      }
	    }
	    else std::cout<<"Method number should be between 1 and 17\n>    ";
	  }
	  else std::cout<<"Method ID not a number\n>    ";
	}

	else if (strValue.compare(0,17,"print processor +")==0){
	  intValue=strValue.substr(17);
	  std::set<Processor>::iterator it;
	  if (is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if ((unsigned)i<all_procs.size()){
	      it = all_procs.begin();
	      std::advance(it, i);
	      std::map<Processor, int>::iterator ite= procs_map.find(*it);
	      if (ite!=procs_map.end() ) procs_map.erase(ite);				
	      pValue=2;
	      procs_map.insert(std::pair<Processor,int>(*it,pValue));
	      procs_map_int.insert(std::pair<int, int>(i, pValue));
	      std::cout<<"The processors added are: ";
	      for (std::map<Processor,int>::const_iterator it =
		  procs_map.begin(); it != procs_map.end(); ++it)
		std::cout<< it->first.id << "   ";
	      std::cout<<"\n>    ";
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";			
	}

	else if (strValue.compare(0,16,"stop processor +")==0){
	  intValue=strValue.substr(16);
	  std::set<Processor>::iterator it;
	  if (is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if ((unsigned)i<all_procs.size()){
	      it = all_procs.begin();
	      std::advance(it, i);
	      std::map<Processor, int>::iterator ite= procs_map.find(*it);
	      if (ite!=procs_map.end()) procs_map.erase(ite);				
	      pValue=1;
	      procs_map.insert(std::pair<Processor,int>(*it,pValue));
	      procs_map_int.insert(std::pair<int, int>(i, pValue));
	      std::cout<<"The processors added are: ";
	      for (std::map<Processor,int>::const_iterator it =
		  procs_map.begin(); it != procs_map.end(); ++it)
		std::cout<< it->first.id << "   ";
	      std::cout<<"\n>    ";
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";			
	}

	/*else if (strValue.compare(0,14,"print memory +")==0){
	  intValue=strValue.substr(14);
	  std::set<Memory>::iterator it;
	  if (is_number(intValue)){
	  int i=std::atoi(intValue.c_str())-1;
	  if((unsigned)i<all_mems.size()){
	  it = all_mems.begin();
	  std::advance(it,i);
	  std::map<Memory, int>::iterator itm = mems_map.find(*it);
	  if (itm!=mems_map.end()) mems_map.erase(itm);
	  pValue=1;									
	  mems_map.insert(std::pair<Memory,int>(*it,pValue));

	  std::cout<<"The memories added are: ";
	  for (std::map<Memory,int>::const_iterator it =
	  mems_map.begin(); it != mems_map.end(); ++it)
	  std::cout<<it->first.id<<"		";
	  std::cout<<"\n>    ";;
	  }
	  else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";
	  }

	  else if (strValue.compare(0,13,"stop memory +")==0){
	  intValue=strValue.substr(13);
	  std::set<Memory>::iterator it;
	  if (is_number(intValue)){
	  int i=std::atoi(intValue.c_str())-1;
	  if((unsigned)i<all_mems.size()){
	  it = all_mems.begin();
	  std::advance(it,i);
	  std::map<Memory, int>::iterator itm = mems_map.find(*it);
	  if (itm!=mems_map.end()) mems_map.erase(itm);
	  pValue=0;									
	  mems_map.insert(std::pair<Memory,int>(*it,pValue));

	  std::cout<<"The memories added are: ";
	  for (std::map<Memory,int>::const_iterator it =
	  mems_map.begin(); it != mems_map.end(); ++it)
	  std::cout<<it->first.id<<"		";
	  std::cout<<"\n>    ";;
	  }
	  else std::cout<<"Invalid number entered\n>    ";
	  }
	  else std::cout<<"Invalid input\n>    ";
	  }*/

	  else if (strValue.compare(0,6,"task -")==0){
	    nameValue=strValue.substr(6);

	    std::vector<std::string>::iterator itp =
	      std::find(print_tasks.begin(), print_tasks.end(),nameValue);  
	    std::vector<std::string>::iterator its =
	      std::find(stop_tasks.begin(),stop_tasks.end(), nameValue);  
	    if (itp!=print_tasks.end())
	    {
	      //tasks_map.erase(it);
	      print_tasks.erase(itp);
	      stop_tasks.erase(its);
	      std::cout<<"The tasks added are: ";
	      for (std::vector<std::string>::const_iterator i =
		  print_tasks.begin(); i != print_tasks.end(); ++i) 
		std::cout<< *i << "  ";
	      std::cout<<"\n>    ";
	    }
	    else{
	      std::cout<<"Task "<<nameValue<<" not present\n";
	      std::cout<<"The tasks added are: ";
	      for (std::vector<std::string>::const_iterator i =
		  print_tasks.begin(); i != print_tasks.end(); ++i)
		std::cout<< *i << "  ";
	      std::cout<<"\n>    ";
	    }
	  }


	  else if (strValue.compare(0,8,"method -")==0){
	    intValue=strValue.substr(8);
	    if(InputNumberCheck(intValue)){
	      Value = std::atoi(intValue.c_str());
	      if (Value>0 && Value<18){
		std::map<int, int>::iterator it = methods_map.find(Value);
		if (it!=methods_map.end()){
		  methods_map.erase(it);				
		  std::cout<<"The methods added are: ";
		  for (std::map<int, int>::const_iterator i =
		      methods_map.begin(); i != methods_map.end(); ++i)
		    std::cout<< function_map[i->first] << "  ";
		  std::cout<<"\n>    ";
		}
		else std::cout<<"Method not present.\n>    "; 
	      }
	      else std::cout<<"Method number should be between 1 and 17\n>    ";
	    }
	    else std::cout<<"Method ID not a number\n>    ";
	  }

	  else if (strValue.compare(0,11,"processor -")==0){
	    intValue=strValue.substr(11);
	    std::set<Processor>::iterator it;
	    if (is_number(intValue)){
	      int i=std::atoi(intValue.c_str())-1;
	      if ((unsigned)i<all_procs.size()){
		it = all_procs.begin();
		std::advance(it, i);
		std::map<Processor, int>::iterator ite= procs_map.find(*it);
		std::map<int, int>::iterator ite_int = procs_map_int.find(i);
		if (ite!=procs_map.end() ){
		  procs_map.erase(ite);
		  procs_map_int.erase(ite_int);				

		  std::cout<<"The processors added are: ";
		  for (std::map<Processor,int>::const_iterator it =
		      procs_map.begin(); it != procs_map.end(); ++it)
		    std::cout<< it->first.id << "   ";
		  std::cout<<"\n>    ";
		}

		else{
		  std::cout<<"Processor not present.\n";
		  std::cout<<"The processors added are: ";
		  for (std::map<Processor,int>::const_iterator it =
		      procs_map.begin(); it != procs_map.end(); ++it)
		    std::cout<< it->first.id << "   ";
		  std::cout<<"\n>    ";
		}
	      }
	      else std::cout<<"Invalid number entered\n>    ";
	    }
	    else std::cout<<"Invalid input\n>    ";

	  }

	  /*else if (strValue.compare(0,8,"memory -")==0){
	    intValue=strValue.substr(8);
	    std::set<Memory>::iterator it;
	    std::map<Memory, int>::iterator ite;
	    if(is_number(intValue)){
	    int i=std::atoi(intValue.c_str())-1;
	    if((unsigned)i<all_mems.size()){
	    it = all_mems.begin();
	    std::advance(it, i);
	    std::map<Memory, int>::iterator ite=mems_map.find(*it);
	    if(ite!=mems_map.end()){
	    mems_map.erase(ite);
	    std::cout<<"The memories added are:	";
	    for (std::map<Memory,int>::const_iterator it =
	    mems_map.begin(); it!=mems_map.end(); ++it)
	    std::cout<<it->first.id<<"		";
	    std::cout<<"\n>    ";
	    }
	    }
	    else std::cout<<"Invalid number entered\n>    ";
	    }
	    else std::cout<<"Invalid input\n>    ";


	    }*/

	    else if (strValue.compare("help")==0){
	      std::cout<<"Following are the commands that can be executed:\n";
	      std::cout<<"task +<task_id> --> To add a task to be monitored \n";
	      std::cout<<"task -<task_id> --> To remove a task from the lists";
	      std::cout<<" of tasks which are being monitored \n";
	      std::cout<<"methods --> To see the list of methods with their ";
	      std::cout<<"corresponding ids\n";
	      std::cout<<"method +<method_id> --> To add a method to be i";
	      std::cout<<"monitored\n";
	      std::cout<<"method -<method_id> --> To remove a method from the ";
	      std::cout<<"lists of methods which are being monitored \n";
	      std::cout<<"processors --> To see the list of processor with ";
	      std::cout<<"their corresponding ids\n";
	      std::cout<<"processor +<processor_id> --> To add a processor to ";
	      std::cout<<"be monitored\n";
	      std::cout<<"processor -<processor_id> --> To remove a processor ";
	      std::cout<<"from the lists of processors which are being ";
	      std::cout<<"monitored \n";
	      std::cout<<">    ";
	    }


	    else if (strValue.compare("methods")==0){
	      for(std::map<int, std::string >::const_iterator it =
		  function_map.begin(); it != function_map.end(); ++it)
	      {
		std::cout << it->first << ". " << it->second << " " << "\n";
	      }
	      std::cout<<">    ";
	    }

	    else if (strValue.compare("processors")==0){
	      int i=0;
	      std::set<Processor>::iterator it;
	      for ( it = all_procs.begin();
		  it != all_procs.end(); it++)
	      {
		i++;
		Processor::Kind k = it->kind();
		if (k == Processor::UTIL_PROC) 
		  std::cout<<i<<". Utility Processor ID:"<<it->id<<"\n";
		else 
		  std::cout<<i<<". Processor ID: "<<it->id<<"  Kind:"<<k<<"\n";
	      }
	      std::cout<<">    ";
	    }
	    /*
	       else if (strValue.compare("memories")==0){
	       int i=0;
	       std::set<Memory>::iterator it;
	       for ( it = all_mems.begin();
	       it != all_mems.end(); it++)
	       {
	       i++;
	       std::cout<<i<<". Memory ID: "<<it->id<<"  Capacity: ";
	       std::cout<<it->capacity()<<"  Kind: "<<it->kind()<<"\n";
	       }
	       std::cout<<">    ";
	       }
	     */
	    else if (strValue.compare("exit")==0){
	      break;
	    }

	    else std::cout<<"Invalid Command\n>    ";

      }
    }

    //Get input to change options set by select_task_options
    void WrapperMapper::get_select_task_options_input
      (const MapperContext ctx, std::string task_name, TaskOptions& output){
	std::string strValue;
	std::cout<<"\nEnter \"change\" to change the list of tasks and ";
	std::cout<<"processors being monitored. Enter \"help\" for the "; 
	std::cout<<"list of commands. Enter \"exit\" to exit\n";
	std::cout<<"\nTo change a task option, enter the the number ";
	std::cout<<"corresponding to the option:\n";
	std::cout<<"1. initial processor\n2. inline task\n3. stealable\n";
	std::cout<<"4. map locally\n>    ";
	while(1){
	  getline(std::cin, strValue);
	  if (strValue.compare("1")==0){
	    int i=0;
	    std::set<Processor>::iterator it;
	    for ( it = all_procs.begin();
		it != all_procs.end(); it++)
	    {
	      i++;
	      Processor::Kind k = it->kind();
	      if (k == Processor::UTIL_PROC)
		std::cout<<i<<". Utility Processor ID:"<<it->id<<"\n";
	      else
		std::cout<<i<<". Processor ID: "<<it->id<<"Kind:"<<k<<"\n";
	    }
	    std::cout<<"Enter the number corresponding to the processor ";
	    std::cout<<"to be selected\n>    ";
	    while(1){
	      std::string strValue1;
	      getline(std::cin, strValue1);
	      if (is_number(strValue1)){
		i=std::atoi(strValue1.c_str())-1;
		if ((unsigned)i<all_procs.size()){
		  it = all_procs.begin();
		  std::advance(it, i);
		  output.initial_proc= *it;
		  std::cout<<"\ninitial processor="<<output.initial_proc.id;
		  std::cout<<"\n";
		  break;
		}
		else std::cout<<"Invalid number entered\n>    ";
	      }
	      else std::cout<<"Invalid input\n>    ";
	    }
	    std::cout<<"\n>    ";
	  }
	  else if (strValue.compare("2")==0){
	    std::cout<<"Enter 0 or 1\n>    ";
	    std::string strValue1;
	    while(1){
	      getline(std::cin, strValue1);
	      if (strValue1=="0"){
		output.inline_task=false;	
		std::cout<<"\ninline task="<<output.inline_task<<"\n";
		break;
	      }
	      else if (strValue1=="1"){
		output.inline_task=true;	
		std::cout<<"\ninline task="<<output.inline_task<<"\n";
		break;
	      }

	      else std::cout<<"Invalid input\n>    ";
	    }
	    std::cout<<"\n>    ";
	  }
	  else if (strValue.compare("3")==0){
	    std::cout<<"Enter 0 or 1\n>    ";
	    std::string strValue1;
	    while(1){
	      getline(std::cin, strValue1);
	      if (strValue1=="0"){
		output.stealable=false;	
		std::cout<<"\nstealable="<<output.stealable<<"\n";
		break;
	      }
	      else if (strValue1=="1"){
		output.stealable=true;	
		std::cout<<"\nstealable="<<output.stealable<<"\n";
		break;
	      }

	      else std::cout<<"Invalid input\n>    ";
	    }
	    std::cout<<"\n>    ";
	  }
	  else if (strValue.compare("4")==0){
	    std::cout<<"Enter 0 or 1\n>    ";
	    std::string strValue1;
	    while(1){
	      getline(std::cin, strValue1);
	      if (strValue1=="0"){
		output.map_locally=false;	
		std::cout<<"\nmap locally="<<output.map_locally<<"\n";
		break;
	      }
	      else if (strValue1=="1"){
		output.map_locally=true;	
		std::cout<<"\nmap locally="<<output.map_locally<<"\n";
		break;
	      }

	      else std::cout<<"Invalid input\n>    ";
	    }
	    std::cout<<"\n>    ";
	  }
	  else if (strValue.compare("change")==0){
	    get_input(ctx);
	    std::cout<<"\n>    ";
	  }
	  else if (strValue.compare("exit")==0) break;
	  else std::cout<<"Invalid input\n>    ";
	}
      }

    const char* WrapperMapper::get_mapper_name(void) const
      //------------------------------------------------------------------------
    {
      return dmapper->get_mapper_name();
    }

    Mapper::MapperSyncModel WrapperMapper::get_mapper_sync_model(void) const
      //------------------------------------------------------------------------
    {
      // Default mapper operates with the serialized re-entrant sync model
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }


    void WrapperMapper::select_task_options(const MapperContext    ctx,
	const Task&            task,
	TaskOptions&     output){

      //Data to be broadcasted the very first time by the owner processor
      if (databroadcasted==0 && node_id==0  && ownerprocessor==local_proc){

	//std::string send_message = Serialize(tasks_map, procs_map_int);
	std::string send_message = 
	  Serialize(print_tasks, stop_tasks, procs_map_int);
	int send_size = send_message.size()+1;
	char send_mess_chars[send_size];
	strcpy(send_mess_chars, send_message.c_str());
	void *message_point = &send_mess_chars;
	mrt->broadcast(ctx, message_point, send_size*sizeof(char));                                        
	databroadcasted=1;

      }

      dmapper->select_task_options(ctx, task, output);

      //Get iterators to the task and processor in the tasks_map and procs_map
      //std::map<std::string, int>::iterator itt = 
      //tasks_map.find(task.get_task_name());

      std::vector<std::string>::iterator ittp = 
	std::find(print_tasks.begin(), print_tasks.end(), task.get_task_name());  
      std::vector<std::string>::iterator itts = 
	std::find(stop_tasks.begin(), stop_tasks.end(), task.get_task_name());  
      std::map<Processor, int>::iterator itp = 
	procs_map.find(output.initial_proc);

      //If owner processor, then communicate with the user, if needed. 
      //If not the owner processor, send the information to the owner processor. 
      if(ownerprocessor==local_proc){

	if (ittp!=print_tasks.end() || itp!=procs_map.end()) {
	  std::cout<<"\n--------------TASK: "<<task.get_task_name();
	  std::cout<<" FUNCTION: select_task_options--------------\n";
	  std::cout<<"\nThe selected task options for task ";
	  std::cout<<task.get_task_name()<<" are as follows:\n";
	  std::cout<<"initial processor="<<output.initial_proc.id;
	  std::cout<<"\ninline task="<<output.inline_task;
	  std::cout<<"\nspawn task="<<output.stealable;
	  std::cout<<"\nmap locally="<<output.map_locally<<"\n\n";
	  if (itts!=stop_tasks.end() || itp->second==1) {
	    std::cout<<"To change the task options, type 'change' and to exit,";
	    std::cout<<" type 'exit'\n";
	    get_select_task_options_input(ctx, task.get_task_name(), output);
	  }
	}
      }
      else if (ittp!=print_tasks.end() || itp!=procs_map.end()) {
	wait_task_options = output;
	int action = 1;
	int task_int = ittp - print_tasks.begin();
	if (itts!=stop_tasks.end() || itp->second==1){
	  action=0;
	  task_int = itts - stop_tasks.begin();
	}
	select_task_options_message message =
	{42356156,task_int,wait_task_options, action};
	void *message_point = &message;
	mapevent = mrt->create_mapper_event(ctx);
	mrt->send_message(ctx,ownerprocessor, message_point, 
	    sizeof(select_task_options_message));
	mrt->wait_on_mapper_event(ctx, mapevent); //Wait for the owner processor
	output = wait_task_options;
      }

    }

    void WrapperMapper::premap_task(const MapperContext      ctx,
	const Task&              task, 
	const PremapTaskInput&   input,
	PremapTaskOutput&        output){
      dmapper->premap_task(ctx, task, input, output);
    }

    void WrapperMapper::slice_task(const MapperContext      ctx,
	const Task&              task, 
	const SliceTaskInput&    input,
	SliceTaskOutput&   output){
      dmapper->slice_task(ctx, task, input, output);
    }

    void WrapperMapper::map_task(const MapperContext      ctx,
	const Task&              task,
	const MapTaskInput&      input,
	MapTaskOutput&     output){
      dmapper->map_task(ctx, task, input, output);
    }

    void WrapperMapper::select_task_variant(const MapperContext          ctx,
	const Task&                  task,
	const SelectVariantInput&    input,
	SelectVariantOutput&   output){
      dmapper->select_task_variant(ctx, task, input, output);
    }

    void WrapperMapper::postmap_task(const MapperContext      ctx,
	const Task&              task,
	const PostMapInput&      input,
	PostMapOutput&     output){
      dmapper->postmap_task(ctx, task, input, output);
    }

    void WrapperMapper::select_task_sources(const MapperContext        ctx,
	const Task&                task,
	const SelectTaskSrcInput&  input,
	SelectTaskSrcOutput& output){
      dmapper->select_task_sources(ctx, task, input, output);
    }

    void WrapperMapper::speculate(const MapperContext      ctx,
	const Task&              task,
	SpeculativeOutput& output){
      dmapper->speculate(ctx, task, output);
    }

    void WrapperMapper::report_profiling(const MapperContext      ctx,
	const Task&              task,
	const TaskProfilingInfo& input){
      dmapper->report_profiling(ctx, task, input);
    }

    void WrapperMapper::map_inline(const MapperContext        ctx,
	const InlineMapping&       inline_op,
	const MapInlineInput&      input,
	MapInlineOutput&     output){
      dmapper->map_inline(ctx, inline_op, input, output);
    }

    void WrapperMapper::select_inline_sources(const MapperContext        ctx,
	const InlineMapping&         inline_op,
	const SelectInlineSrcInput&  input,
	SelectInlineSrcOutput& output){
      dmapper->select_inline_sources(ctx, inline_op, input, output);
    }

    void WrapperMapper::report_profiling(const MapperContext         ctx,
	const InlineMapping&        inline_op,
	const InlineProfilingInfo&  input){
      dmapper->report_profiling(ctx, inline_op, input);
    }

    void WrapperMapper::map_copy(const MapperContext      ctx,
	const Copy&              copy,
	const MapCopyInput&      input,
	MapCopyOutput&     output){
      dmapper->map_copy(ctx, copy, input, output);
    }

    void WrapperMapper::select_copy_sources(const MapperContext          ctx,
	const Copy&                  copy,
	const SelectCopySrcInput&    input,
	SelectCopySrcOutput&   output){
      dmapper->select_copy_sources(ctx, copy, input, output);
    }

    void WrapperMapper::speculate(const MapperContext      ctx,
	const Copy& copy,
	SpeculativeOutput& output){
      dmapper->speculate(ctx, copy, output);
    }

    void WrapperMapper::report_profiling(const MapperContext      ctx,
	const Copy&              copy,
	const CopyProfilingInfo& input){
      dmapper->report_profiling(ctx, copy, input);
    }

    void WrapperMapper::select_close_sources(const MapperContext        ctx,
	const Close&               close,
	const SelectCloseSrcInput&  input,
	SelectCloseSrcOutput& output){
      dmapper->select_close_sources(ctx, close, input, output);
    }

    void WrapperMapper::report_profiling(const MapperContext       ctx,
	const Close&              close,
	const CloseProfilingInfo& input){
      dmapper->report_profiling(ctx, close, input);
    }

    void WrapperMapper::map_acquire(const MapperContext         ctx,
	const Acquire&              acquire,
	const MapAcquireInput&      input,
	MapAcquireOutput&     output){
      dmapper->map_acquire(ctx, acquire, input, output);
    }

    void WrapperMapper::speculate(const MapperContext         ctx,
	const Acquire&              acquire,
	SpeculativeOutput&    output){
      dmapper->speculate(ctx, acquire, output);			
    }

    void WrapperMapper::report_profiling(const MapperContext         ctx,
	const Acquire&              acquire,
	const AcquireProfilingInfo& input){
      dmapper->report_profiling(ctx, acquire, input);
    }

    void WrapperMapper::map_release(const MapperContext         ctx,
	const Release&              release,
	const MapReleaseInput&      input,
	MapReleaseOutput&     output){
      dmapper->map_release(ctx, release, input, output);
    }

    void WrapperMapper::select_release_sources(const MapperContext       ctx,
	const Release&                 release,
	const SelectReleaseSrcInput&   input,
	SelectReleaseSrcOutput&  output){
      dmapper->select_release_sources(ctx, release, input, output);
    }

    void WrapperMapper::speculate(const MapperContext         ctx,
	const Release&              release,
	SpeculativeOutput&    output){
      dmapper->speculate(ctx, release, output);
    }

    void WrapperMapper::report_profiling(const MapperContext         ctx,
	const Release&              release,
	const ReleaseProfilingInfo& input){
      dmapper->report_profiling(ctx, release, input);
    }

    void WrapperMapper::configure_context(const MapperContext         ctx,
	const Task&                 task,
	ContextConfigOutput&  output){
      dmapper->configure_context(ctx, task, output);
    }

    void WrapperMapper::select_tunable_value(const MapperContext         ctx,
	const Task&                 task,
	const SelectTunableInput&   input,
	SelectTunableOutput&  output){
      dmapper->select_tunable_value(ctx, task, input, output);
    }

    void WrapperMapper::map_must_epoch(const MapperContext           ctx,
	const MapMustEpochInput&      input,
	MapMustEpochOutput&     output){
      dmapper->map_must_epoch(ctx, input, output);
    }

    void WrapperMapper::map_dataflow_graph(const MapperContext           ctx,
	const MapDataflowGraphInput&  input,
	MapDataflowGraphOutput& output){
      dmapper->map_dataflow_graph(ctx, input, output);
    }

    void WrapperMapper::select_tasks_to_map(const MapperContext          ctx,
	const SelectMappingInput&    input,
	SelectMappingOutput&   output){
      dmapper->select_tasks_to_map(ctx, input, output);
    }

    void WrapperMapper::select_steal_targets(const MapperContext         ctx,
	const SelectStealingInput&  input,
	SelectStealingOutput& output){
      dmapper->select_steal_targets(ctx, input, output);
    }

    void WrapperMapper::permit_steal_request(const MapperContext         ctx,
	const StealRequestInput&    input,
	StealRequestOutput&   output){
      dmapper->permit_steal_request(ctx, input, output);
    }

    void WrapperMapper::handle_message(const MapperContext           ctx,
	const MapperMessage&          message){
      const select_task_options_message *rec_message = 
	(select_task_options_message*)message.message;

      if (node_id==0 && ownerprocessor.id==local_proc.id){
	if (rec_message->tag==42356156){
	  //Owner processor gets a message with the tag, so communicate with the user
	  int task_int = rec_message->task_name;
	  TaskOptions output = rec_message->output;
	  int action = rec_message->action;
	  std::string task_name;
	  if (action && (unsigned)task_int<print_tasks.size()) 
	    task_name = print_tasks[task_int]; 
	  else if (!action && (unsigned)task_int<stop_tasks.size()) 
	    task_name = stop_tasks[task_int];
	  else action=2;
	  if (action==1 || action==0){
	    std::cout<<"\n--------------TASK: "<<task_name;
	    std::cout<<" FUNCTION: select_task_options--------------\n";
	    std::cout<<"\nThe selected task options for task "<<task_name;
	    std::cout<<" are as follows:\n";
	    std::cout<<"initial processor="<<output.initial_proc.id;
	    std::cout<<"\ninline task="<<output.inline_task;
	    std::cout<<"\nspawn task="<<output.stealable;
	    std::cout<<"\nmap locally="<<output.map_locally<<"\n\n";
	  }
	  if (!action){
	    std::cout<<"To change the task options, type 'change' and to exit,";
	    std::cout<<" type 'exit'\n";
	    get_select_task_options_input(ctx, task_name, output);
	  }
	  select_task_options_message mess ={42356156,task_int,output, action};
	  void *message_point = &mess;
	  mrt->send_message(ctx,message.sender, message_point, 
	      sizeof(select_task_options_message));
	}
      }
      //Message from owner processor, so trigger the wait event
      else if (rec_message->tag ==42356156){
	wait_task_options = rec_message->output;				
	mrt->trigger_mapper_event(ctx, mapevent);
      }

      //This is the broadcast message and so, deserialize the message.
      else {
	const char *rec1_message =(const char *)message.message;

	if (node_id!=0 && localowner == local_proc){	
	  std::string rec_string = rec1_message;		
	  Deserialize(rec_string);	
	}
      }

    }	


    void WrapperMapper::handle_task_result(const MapperContext           ctx,
	const MapperTaskResult&       result){
      dmapper->handle_task_result(ctx, result);
    }
  };
};
