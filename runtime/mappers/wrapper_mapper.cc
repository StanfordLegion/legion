#include <cstdlib>
#include <cassert>
#include <limits.h>
#include <algorithm>

#include "wrapper_mapper.h"


#define STATIC_MAX_PERMITTED_STEALS   4
#define STATIC_MAX_STEAL_COUNT        2
#define STATIC_BREADTH_FIRST          false
#define STATIC_STEALING_ENABLED       false
#define STATIC_MAX_SCHEDULE_COUNT     8


namespace Legion {
  namespace Mapping{

		std::vector<int> WrapperMapper::tasks_list;
		std::vector<int> WrapperMapper::functions_list;
		std::set<Memory> WrapperMapper::all_mems;
		std::set<Processor> WrapperMapper::all_procs;
		std::vector<Memory> WrapperMapper::mems_list;
		std::vector<Processor> WrapperMapper::procs_list;
		std::map<int, int> WrapperMapper::methods_map;
		std::map<Processor, int> WrapperMapper::procs_map;
		std::map<std::string, int> WrapperMapper::tasks_map;
		std::map<Memory, int> WrapperMapper::mems_map;
		std::map<int, std::string> WrapperMapper::task_names_map;
		bool WrapperMapper::inputtaken=0;
		Processor WrapperMapper::ownerprocessor;
		int WrapperMapper::messagecount = 0;
		MapperEvent WrapperMapper::mapevent;			
		
		WrapperMapper::WrapperMapper(Mapper* dmapper, Machine machine, Processor local):Mapper(), dmapper(dmapper), local_proc(local), local_kind(local.kind()), 
        node_id(local.address_space()), machine(machine),
        max_steals_per_theft(STATIC_MAX_PERMITTED_STEALS),
        max_steal_count(STATIC_MAX_STEAL_COUNT),
        breadth_first_traversal(STATIC_BREADTH_FIRST),
        stealing_enabled(STATIC_STEALING_ENABLED),
        max_schedule_count(STATIC_MAX_SCHEDULE_COUNT){
			machine.get_all_processors(WrapperMapper::all_procs);
			machine.get_all_memories(WrapperMapper::all_mems);
			if (!WrapperMapper::inputtaken){
				WrapperMapper::get_input();
				WrapperMapper::inputtaken=1;
				WrapperMapper::ownerprocessor = local;
			}

//for(std::map<int, std::string>::const_iterator it = task_names_map.begin();
//    it != task_names_map.end(); ++it)
//{
//    std::cout << it->first << " " << it->second<< "\n";
//}
		}
		WrapperMapper::~WrapperMapper(){
		}

		bool is_number(const std::string& s)
		{
			std::string::const_iterator it = s.begin();
			while (it != s.end() && std::isdigit(*it)) ++it;
			return !s.empty() && it == s.end();
		}

		bool is_valid_name(const std::string& s){
			std::map<int,std::string>::iterator it = WrapperMapper::task_names_map.begin();
			while (it!=WrapperMapper::task_names_map.end()){
				if (s==it->second) break;
				++it;
			}
			if (it!=WrapperMapper::task_names_map.end()) return 1;
			else return 0;
		}
		bool WrapperMapper::InputNumberCheck(std::string strUserInput)
		{
			for (unsigned int nIndex=0; nIndex < strUserInput.length(); nIndex++)
			{
				if (!std::isdigit(strUserInput[nIndex])) return false;
			}
			return 1;
		}

		void WrapperMapper::get_input(){
			std::string strValue;
			std::map<int, std::string> function_map;
			int Value, pValue;

			function_map[1] = "select_task_options"; function_map[2] = "select_tasks_to_schedule";
			function_map[3] = "target_task_steal"; function_map[4] = "permit_task_steal";
			function_map[5] = "slice_domain"; function_map[6] = "pre_map_task";
			function_map[7] = "select_task_variant"; function_map[8] = "map_task";
			function_map[9] = "post_map_task"; function_map[10] = "map_copy";
			function_map[11] = "map_inline"; function_map[12] = "map_must_epoch";
			function_map[13] = "notify_mapping_result"; function_map[14] = "notify_mapping_failed";
			function_map[15] = "rank_copy_targets"; function_map[16] = "rank_copy_sources";
			function_map[17] = "Other";

			std::cout<< "Type 'help' to see the list of commands. Type 'exit' to exit.\n";
			std::cout<<">    ";
			while (1)
			{
				getline(std::cin, strValue); 
				std::string nameValue;
				std::string intValue;
				if (strValue.compare(0,12,"print task +")==0){
					nameValue=strValue.substr(12);
					//if(is_valid_name(nameValue)){
						std::map<std::string, int>::iterator it = WrapperMapper::tasks_map.find(nameValue);
						if (it==WrapperMapper::tasks_map.end())
						{
							pValue=1;
							WrapperMapper::tasks_map.insert(std::pair<std::string, int>(nameValue,pValue));
							std::cout<<"The tasks added are: ";
							for (std::map<std::string, int>::const_iterator i = WrapperMapper::tasks_map.begin(); i != WrapperMapper::tasks_map.end(); ++i) std::cout<< i->first << "  ";
							std::cout<<"\n>    ";
						}
						else{
							WrapperMapper::tasks_map.erase(it);
							pValue=1;
							WrapperMapper::tasks_map.insert(std::pair<std::string, int>(nameValue,pValue));
							std::cout<<"The tasks added are: ";
							for (std::map<std::string, int>::const_iterator i = WrapperMapper::tasks_map.begin(); i != WrapperMapper::tasks_map.end(); ++i) std::cout<< i->first << "  ";
							std::cout<<"\n>    ";
						}
					//}
					//else std::cout<<"No task of that name\n>    ";
				}

				else if (strValue.compare(0,11,"stop task +")==0){
					nameValue=strValue.substr(11);
				//	if(is_valid_name(nameValue)){
						std::map<std::string, int>::iterator it = WrapperMapper::tasks_map.find(nameValue);
						if (it==WrapperMapper::tasks_map.end())
						{
							pValue=0;
							WrapperMapper::tasks_map.insert(std::pair<std::string, int>(nameValue,pValue));
							std::cout<<"The tasks added are: ";
							for (std::map<std::string, int>::const_iterator i = WrapperMapper::tasks_map.begin(); i != WrapperMapper::tasks_map.end(); ++i) std::cout<< i->first << "  ";
							std::cout<<"\n>    ";
						}
						else{
							WrapperMapper::tasks_map.erase(it);
							pValue=0;
							WrapperMapper::tasks_map.insert(std::pair<std::string, int>(nameValue,pValue));
							std::cout<<"The tasks added are: ";
							for (std::map<std::string, int>::const_iterator i = WrapperMapper::tasks_map.begin(); i != WrapperMapper::tasks_map.end(); ++i) std::cout<< i->first << "  ";
							std::cout<<"\n>    ";
						}
				//	}
				//	else std::cout<<"No task of that name\n>    ";
				}

				else if (strValue.compare(0,14,"print method +")==0){
					intValue=strValue.substr(14);
					if(InputNumberCheck(intValue)){
						Value = std::atoi(intValue.c_str());
						if (Value>0 && Value<18){ 
							std::map<int, int>::iterator it = WrapperMapper::methods_map.find(Value);
							if (it==WrapperMapper::methods_map.end()){
								pValue=1;
								WrapperMapper::methods_map.insert(std::pair<int, int>(Value,pValue));
								std::cout<<"The methods added are: ";
								for (std::map<int, int>::const_iterator i = WrapperMapper::methods_map.begin(); i != WrapperMapper::methods_map.end(); ++i) std::cout<< function_map[i->first] << "  ";
								std::cout<<"\n>    ";
							}
							else{
								WrapperMapper::methods_map.erase(it);
								pValue=1;
								WrapperMapper::methods_map.insert(std::pair<int, int>(Value,pValue));
								std::cout<<"The methods added are: ";
								for (std::map<int, int>::const_iterator i = WrapperMapper::methods_map.begin(); i != WrapperMapper::methods_map.end(); ++i) std::cout<< function_map[i->first] << "  ";
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
							std::map<int, int>::iterator it = WrapperMapper::methods_map.find(Value);
							if (it==WrapperMapper::methods_map.end()){
								pValue=0;
								WrapperMapper::methods_map.insert(std::pair<int, int>(Value,pValue));
								std::cout<<"The methods added are: ";
								for (std::map<int, int>::const_iterator i = WrapperMapper::methods_map.begin(); i != WrapperMapper::methods_map.end(); ++i) std::cout<< function_map[i->first] << "  ";
								std::cout<<"\n>    ";
							}
							else{
								WrapperMapper::methods_map.erase(it);
								pValue=0;
								WrapperMapper::methods_map.insert(std::pair<int, int>(Value,pValue));
								std::cout<<"The methods added are: ";
								for (std::map<int, int>::const_iterator i = WrapperMapper::methods_map.begin(); i != WrapperMapper::methods_map.end(); ++i) std::cout<< function_map[i->first] << "  ";
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
						if ((unsigned)i<WrapperMapper::all_procs.size()){
							it = WrapperMapper::all_procs.begin();
							std::advance(it, i);
							std::map<Processor, int>::iterator ite= WrapperMapper::procs_map.find(*it);
							if (ite!=WrapperMapper::procs_map.end() ) WrapperMapper::procs_map.erase(ite);				
							pValue=1;
							WrapperMapper::procs_map.insert(std::pair<Processor,int>(*it,pValue));
							
							std::cout<<"The processors added are: ";
							for (std::map<Processor,int>::const_iterator it = WrapperMapper::procs_map.begin(); it != WrapperMapper::procs_map.end(); ++it) std::cout<< it->first.id << "   ";
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
						if ((unsigned)i<WrapperMapper::all_procs.size()){
							it = WrapperMapper::all_procs.begin();
							std::advance(it, i);
							std::map<Processor, int>::iterator ite= WrapperMapper::procs_map.find(*it);
							if (ite!=WrapperMapper::procs_map.end()) WrapperMapper::procs_map.erase(ite);				
							pValue=0;
							WrapperMapper::procs_map.insert(std::pair<Processor,int>(*it,pValue));
							
							std::cout<<"The processors added are: ";
							for (std::map<Processor,int>::const_iterator it = WrapperMapper::procs_map.begin(); it != WrapperMapper::procs_map.end(); ++it) std::cout<< it->first.id << "   ";
							std::cout<<"\n>    ";
						}
						else std::cout<<"Invalid number entered\n>    ";
					}
					else std::cout<<"Invalid input\n>    ";			
				}
				
				else if (strValue.compare(0,14,"print memory +")==0){
					intValue=strValue.substr(14);
					std::set<Memory>::iterator it;
					if (is_number(intValue)){
						int i=std::atoi(intValue.c_str())-1;
						if((unsigned)i<WrapperMapper::all_mems.size()){
							it = WrapperMapper::all_mems.begin();
							std::advance(it,i);
							std::map<Memory, int>::iterator itm = WrapperMapper::mems_map.find(*it);
							if (itm!=WrapperMapper::mems_map.end()) WrapperMapper::mems_map.erase(itm);
							pValue=1;									
							WrapperMapper::mems_map.insert(std::pair<Memory,int>(*it,pValue));
							
							std::cout<<"The memories added are: ";
							for (std::map<Memory,int>::const_iterator it = WrapperMapper::mems_map.begin(); it != WrapperMapper::mems_map.end(); ++it) std::cout<<it->first.id<<"		";
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
						if((unsigned)i<WrapperMapper::all_mems.size()){
							it = WrapperMapper::all_mems.begin();
							std::advance(it,i);
							std::map<Memory, int>::iterator itm = WrapperMapper::mems_map.find(*it);
							if (itm!=WrapperMapper::mems_map.end()) WrapperMapper::mems_map.erase(itm);
							pValue=0;									
							WrapperMapper::mems_map.insert(std::pair<Memory,int>(*it,pValue));
							
							std::cout<<"The memories added are: ";
							for (std::map<Memory,int>::const_iterator it = WrapperMapper::mems_map.begin(); it != WrapperMapper::mems_map.end(); ++it) std::cout<<it->first.id<<"		";
							std::cout<<"\n>    ";;
						}
						else std::cout<<"Invalid number entered\n>    ";
					}
					else std::cout<<"Invalid input\n>    ";
				}
				else if (strValue.compare(0,6,"task -")==0){
					nameValue=strValue.substr(6);
					//if(is_valid_name(nameValue)){
						std::map<std::string, int>::iterator it = WrapperMapper::tasks_map.find(nameValue);
						if (it!=WrapperMapper::tasks_map.end())
						{WrapperMapper::tasks_map.erase(it);
							std::cout<<"The tasks added are: ";
							for (std::map<std::string, int>::const_iterator i = WrapperMapper::tasks_map.begin(); i != WrapperMapper::tasks_map.end(); ++i) std::cout<< i->first << "  ";
							std::cout<<"\n>    ";
						}
						else std::cout<<"Task "<<Value<<" not present\n>    ";
					//}
					//else std::cout<<"Task ID not a number\n>    ";
				}
				

				else if (strValue.compare(0,8,"method -")==0){
					intValue=strValue.substr(8);
					if(InputNumberCheck(intValue)){
						Value = std::atoi(intValue.c_str());
						if (Value>0 && Value<18){
							std::map<int, int>::iterator it = WrapperMapper::methods_map.find(Value);
							if (it!=WrapperMapper::methods_map.end()){
								WrapperMapper::methods_map.erase(it);				
								std::cout<<"The methods added are: ";
								for (std::map<int, int>::const_iterator i = WrapperMapper::methods_map.begin(); i != WrapperMapper::methods_map.end(); ++i) std::cout<< function_map[i->first] << "  ";
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
					std::map<Processor, int>::iterator ite;
					if (is_number(intValue)){
						int i=std::atoi(intValue.c_str())-1;
						if ((unsigned)i<WrapperMapper::all_procs.size()){
							it = WrapperMapper::all_procs.begin();
							std::advance(it, i);
							std::map<Processor, int>::iterator ite= WrapperMapper::procs_map.find(*it);
							if (ite!=WrapperMapper::procs_map.end() ){
								WrapperMapper::procs_map.erase(ite);				
								std::cout<<"The processors added are: ";
								for (std::map<Processor,int>::const_iterator it = WrapperMapper::procs_map.begin(); it != WrapperMapper::procs_map.end(); ++it) std::cout<< it->first.id << "   ";
								std::cout<<"\n>    ";
							}
						}
						else std::cout<<"Invalid number entered\n>    ";
					}
					else std::cout<<"Invalid input\n>    ";

				}
				
				else if (strValue.compare(0,8,"memory -")==0){
					intValue=strValue.substr(8);
					std::set<Memory>::iterator it;
					std::map<Memory, int>::iterator ite;
					if(is_number(intValue)){
						int i=std::atoi(intValue.c_str())-1;
						if((unsigned)i<WrapperMapper::all_mems.size()){
							it = WrapperMapper::all_mems.begin();
							std::advance(it, i);
							std::map<Memory, int>::iterator ite=WrapperMapper::mems_map.find(*it);
							if(ite!=WrapperMapper::mems_map.end()){
								WrapperMapper::mems_map.erase(ite);
								std::cout<<"The memories added are:	";
								for (std::map<Memory,int>::const_iterator it = WrapperMapper::mems_map.begin(); it!=WrapperMapper::mems_map.end(); ++it) std::cout<<it->first.id<<"		";
								std::cout<<"\n>    ";
							}
						}
						else std::cout<<"Invalid number entered\n>    ";
					}
					else std::cout<<"Invalid input\n>    ";
					
					
				}
				
				else if (strValue.compare("help")==0){
					std::cout<<"Following are the commands that can be executed:\n";
					std::cout<<"task +<task_id> --> To add a task to be monitored \n";
					std::cout<<"task -<task_id> --> To remove a task from the lists of tasks which are being monitored \n";
					std::cout<<"methods --> To see the list of methods with their corresponding ids\n";
					std::cout<<"method +<method_id> --> To add a method to be monitored\n";
					std::cout<<"method -<method_id> --> To remove a method from the lists of methods which are being monitored \n";
					std::cout<<"processors --> To see the list of processor with their corresponding ids\n";
					std::cout<<"processor +<processor_id> --> To add a processor to be monitored\n";
					std::cout<<"processor -<processor_id> --> To remove a processor from the lists of processors which are being monitored \n";
					std::cout<<">    ";
				}

				else if (strValue.compare("tasks")==0){
					std::cout<<"Task ID"<<"	"<<"Task Name"<<"\n";
					for (std::map<int, std::string>::iterator it = WrapperMapper::task_names_map.begin(); it!=WrapperMapper::task_names_map.end(); it++) std::cout<<it->first<<"	"<<it->second<<"\n";
					std::cout<<">    ";
				}				

				else if (strValue.compare("methods")==0){
					for(std::map<int, std::string >::const_iterator it = function_map.begin(); it != function_map.end(); ++it)
					{
						std::cout << it->first << ". " << it->second << " " << "\n";
					}
					std::cout<<">    ";
				}
				
				else if (strValue.compare("processors")==0){
					int i=0;
					std::set<Processor>::iterator it;
					for ( it = WrapperMapper::all_procs.begin();
					it != WrapperMapper::all_procs.end(); it++)
					{
						i++;
						Processor::Kind k = it->kind();
						if (k == Processor::UTIL_PROC) std::cout<<i<<". Utility Processor ID:"<<it->id<<"\n";
						else std::cout<<i<<". Processor ID: "<<it->id<<"  Kind:"<<k<<"\n";
					}
					std::cout<<">    ";
				}
				
				else if (strValue.compare("memories")==0){
					int i=0;
					std::set<Memory>::iterator it;
					for ( it = WrapperMapper::all_mems.begin();
					it != WrapperMapper::all_mems.end(); it++)
					{
						i++;
						std::cout<<i<<". Memory ID: "<<it->id<<"  Capacity: "<<it->capacity()<<"  Kind: "<<it->kind()<<"\n";
					}
					std::cout<<">    ";
				}
				
				else if (strValue.compare("exit")==0) break;
				
				else std::cout<<"Invalid Command\n>    ";
				
			}
		}

		void WrapperMapper::get_select_task_options_input(const Task& task, TaskOptions& output){
			std::string strValue;
			std::cout<<"\nType change to change the list of tasks and methods being monitored. Type help for the list of commands. Type exit to exit\n";
			std::cout<<"\nTo change a task option, enter the the number corresponding to the option:\n";
			std::cout<<"1. initial processor\n2. inline task\n3. stealable\n4. map locally\n>    ";
			while(1){
				getline(std::cin, strValue);
				if (strValue.compare("1")==0){
					int i=0;
					std::set<Processor>::iterator it;
					for ( it = WrapperMapper::all_procs.begin();
					it != WrapperMapper::all_procs.end(); it++)
					{
						i++;
						Processor::Kind k = it->kind();
						if (k == Processor::UTIL_PROC)
						std::cout<<i<<". Utility Processor ID:"<<it->id<<"\n";
						else
						std::cout<<i<<". Processor ID: "<<it->id<<"Kind:"<<k<<"\n";
					}
					std::cout<<"Enter the number corresponding to the processor to be selected\n>    ";
					while(1){
					std::string strValue1;
						getline(std::cin, strValue1);
						if (is_number(strValue1)){
							i=std::atoi(strValue1.c_str())-1;
				if ((unsigned)i<WrapperMapper::all_procs.size()){
								it = WrapperMapper::all_procs.begin();
								std::advance(it, i);
								output.initial_proc= *it;
								std::cout<<"\ninitial processor="<<output.initial_proc.id<<"\n";
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
					WrapperMapper::get_input();
					std::cout<<"\n>    ";
				}
				else if (strValue.compare("exit")==0) break;
				else std::cout<<"Invalid input\n>    ";
			}
		}

		void WrapperMapper::get_map_task_input(Task *task){
			std::string strValue;
			std::cout<< "Type change to change the list of tasks and methods being monitored. Type 'exit' to exit.\n>    ";			
			while (1)
			{
				getline(std::cin, strValue); 
				if (strValue.compare("change")==0){
					WrapperMapper::get_input();
					std::cout<<"\n>    ";
				}
				else if (strValue.compare("exit")==0) break;
				else std::cout<<"Invalid input\n>    ";
			}
		}
const char* WrapperMapper::get_mapper_name(void) const
    //--------------------------------------------------------------------------
    {
      return dmapper->get_mapper_name();
    }

    Mapper::MapperSyncModel WrapperMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      // Default mapper operates with the serialized re-entrant sync model
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

	struct select_task_options_message{
	const Task& task;
	Mapper::TaskOptions& output;
	};
void WrapperMapper::select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output){


//if(WrapperMapper::ownerprocessor!=local_proc){
//int message = ++WrapperMapper::messagecount;
//void *mess_point;
//mess_point=&message;
//std::cout<<"Message sent \n";
//WrapperMapper::mapevent = mapper_rt_create_mapper_event(ctx);
//std::cout<<"Rajshah1"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";
//WrapperMapper::mapper_rt_send_message(ctx,WrapperMapper::ownerprocessor, mess_point, sizeof(int));
//std::cout<<"Rajshah2"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";
//mapper_rt_wait_on_mapper_event(ctx, WrapperMapper::mapevent);
//}
			dmapper->select_task_options(ctx, task, output);
std::map<std::string, int>::iterator itt = WrapperMapper::tasks_map.find(task.get_task_name());
			std::map<int, int>::iterator itm = WrapperMapper::methods_map.find(1);
			std::map<Processor, int>::iterator itp = WrapperMapper::procs_map.find(output.initial_proc);
			if (((itt!=WrapperMapper::tasks_map.end()) && (itm!=WrapperMapper::methods_map.end())) || (itp!=WrapperMapper::procs_map.end())) {
		if(WrapperMapper::ownerprocessor==local_proc){
				std::cout<<"\n--------------TASK: "<<task.get_task_name()<<" FUNCTION: select_task_options--------------\n";
				std::cout<<"\nThe selected task options for task "<<task.get_task_name()<<" are as follows:\n";
				std::cout<<"initial processor="<<output.initial_proc.id<<"\ninline task="<<output.inline_task;
				std::cout<<"\nspawn task="<<output.stealable<<"\nmap locally="<<output.map_locally<<"\n\n";
				if (((itt!=tasks_map.end() && itt->second==0) && (itm!=methods_map.end() && itm->second==0))||(itp!=procs_map.end() && itp->second==0)) {
					std::cout<<"To change the task options, type 'change' and to exit, type 'exit'\n";
					WrapperMapper::get_select_task_options_input(task, output);
				}
				}
			else{

				select_task_options_message message ={task,output};
				void *message_point = &message;
				//std::cout<<"Message sent \n";
				WrapperMapper::mapevent = mapper_rt_create_mapper_event(ctx);
				WrapperMapper::mapper_rt_send_message(ctx,WrapperMapper::ownerprocessor, message_point, sizeof(select_task_options_message));
mapper_rt_wait_on_mapper_event(ctx, WrapperMapper::mapevent);
				}
			std::cout<<"Test processor"<<output.initial_proc.id<<"\n";
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

void WrapperMapper::map_close(const MapperContext       ctx,
                             const Close&              close,
                             const MapCloseInput&      input,
                                   MapCloseOutput&     output){
			dmapper->map_close(ctx, close, input, output);
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
			if (local_proc==WrapperMapper::ownerprocessor){
//std::cout<<"Owner mapper recieved a message"<<*(int*)message.message<<"\n";
select_task_options_message *rec_message = (select_task_options_message*)message.message;
//std::cout<<"Owner mapper recieved a message"<<rec_message->task.get_task_name();
const Task& task = rec_message->task;
TaskOptions& output = rec_message->output;

std::map<std::string, int>::iterator itt = WrapperMapper::tasks_map.find(task.get_task_name());
			std::map<int, int>::iterator itm = WrapperMapper::methods_map.find(1);
			std::map<Processor, int>::iterator itp = WrapperMapper::procs_map.find(output.initial_proc);
			if (((itt!=WrapperMapper::tasks_map.end()) && (itm!=WrapperMapper::methods_map.end())) || (itp!=WrapperMapper::procs_map.end())) {
				std::cout<<"\n--------------TASK: "<<task.get_task_name()<<" FUNCTION: select_task_options--------------\n";
				std::cout<<"\nThe selected task options for task "<<task.get_task_name()<<" are as follows:\n";
				std::cout<<"initial processor="<<output.initial_proc.id<<"\ninline task="<<output.inline_task;
				std::cout<<"\nspawn task="<<output.stealable<<"\nmap locally="<<output.map_locally<<"\n\n";
				if (((itt!=tasks_map.end() && itt->second==0) && (itm!=methods_map.end() && itm->second==0))||(itp!=procs_map.end() && itp->second==0)) {
					std::cout<<"To change the task options, type 'change' and to exit, type 'exit'\n";
					WrapperMapper::get_select_task_options_input(task, output);
				}
			}

int message1 = ++WrapperMapper::messagecount;
void *mess_point;
mess_point=&message1;
//std::string input;
//while(1){
//			getline(std::cin, input);
//			if (input.compare("go")==0) break;
//}
//std::cout<<"Message sent by owner\n";
WrapperMapper::mapper_rt_send_message(ctx,message.sender, mess_point, sizeof(int));
}
else{
//std::cout<<"Rajshah3"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";
std::cout<<"Message recieved by mapper \n";}
mapper_rt_trigger_mapper_event(ctx, WrapperMapper::mapevent);
//std::cout<<"Rajshah4"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";

//std::string input;
//std::cout<<"Rajshah3"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";
//mapper_rt_trigger_mapper_event(ctx, WrapperMapper::mapevent);
//std::cout<<"Rajshah4"<<mapper_rt_has_mapper_event_triggered(ctx, WrapperMapper::mapevent)<<"\n";
//			while(1){
//			getline(std::cin, input);
//			if (input.compare("go")==0){
//			mapper_rt_trigger_mapper_event(ctx, WrapperMapper::mapevent);
//			break;
//}
//else std::cout<<"INvalid input\n";
//}
//			dmapper->handle_message(ctx, message);
}
void WrapperMapper::handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result){
			dmapper->handle_task_result(ctx, result);
}
};
};
