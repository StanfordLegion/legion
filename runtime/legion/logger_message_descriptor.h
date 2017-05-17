//
//  logger_message_descriptor.h
//  Legion
//
//  Created by Alan Heirich on 5/5/17.
//
//

#ifndef LoggerMessageDescriptor_h
#define LoggerMessageDescriptor_h

#include "realm/logging.h"

namespace Legion{
    namespace Internal {
        const class MessageDescriptor {
        public:
            MessageDescriptor() : mID(0), mDescription(nullptr) {}
            MessageDescriptor(Realm::LoggerMessageID id, std::string htmlDescription){
                mID = id;
#ifdef LOG_MESSAGE_KEEP_DESCRIPTION
                mDescription = htmlDescription;
#endif
            }
            Realm::LoggerMessageID id() const{ return mID; }
            
            std::string description() const{
#ifdef LOG_MESSAGE_KEEP_DESCRIPTION
                return mDescription;
#else
                return "LoggerMessageDescriptor description not saved, recompile with LOG_MESSAGE_KEEP_DESCRIPTION to save it";
#endif
            }
            
            std::string formattedOutput() const {
                return "{ " + std::to_string(mID) + ", \"" + description() + "\" },";
            }
            
            std::string operator << (const std::string &t) { return std::to_string(mID) + t; }
            
        private:
            Realm::LoggerMessageID mID;
            std::string mDescription;
        };
    }
}



#endif /* logger_message_descriptor.h */
