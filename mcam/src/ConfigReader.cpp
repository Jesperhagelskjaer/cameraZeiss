// ConfigReader.cpp
// Class for reading named values from configuration files
// Richard J. Wagner  v2.1  24 May 2004  wagnerr@umich.edu

// Copyright (c) 2004 Richard J. Wagner

#include <cstdlib>

#include <ConfigReader.hpp>

using std::string;

ConfigReader::ConfigReader(string filename, string delimiter, string comment, string sentry) :
    myDelimiter(delimiter), myComment(comment), mySentry(sentry)
{
    // Construct a ConfigReader, getting keys and values from given file

    // Logger logger = util::Logger::getDefault()("default", false);


        string configFileName = filename;
        std::ifstream in(configFileName.c_str());
        if (in) {
            // logger.info("Read config file: %s", configFileName.c_str());
            in >> (*this);
        } else {
            // logger.warn("Config file not available: filename=%s", configFileName.c_str());
        }

}

ConfigReader::ConfigReader() :
    myDelimiter(string(1, '=')), myComment(string(1, '#'))
{
    // Construct a ConfigReader without a file; empty
}

void ConfigReader::remove(const string& key)
{
    // Remove key and its value
    myContents.erase(myContents.find(key));
    return;
}

bool ConfigReader::keyExists(const string& key) const
{
    // Indicate whether key is found
    mapci p = myContents.find(key);
    return (p != myContents.end());
}

/* static */
void ConfigReader::trim(string& s)
{
    // Remove leading and trailing whitespace
    static const char whitespace[] = " \n\t\v\r\f";
    s.erase(0, s.find_first_not_of(whitespace));
    s.erase(s.find_last_not_of(whitespace) + 1U);
}

std::ostream& operator<<(std::ostream& os, const ConfigReader& cf)
{
    // Save a ConfigReader to os
    for (ConfigReader::mapci p = cf.myContents.begin(); p != cf.myContents.end(); ++p)
    {
        os << p->first << " " << cf.myDelimiter << " ";
        os << p->second << std::endl;
    }
    return os;
}

std::istream& operator>>(std::istream& is, ConfigReader& cf)
{
    // Load a ConfigReader from is
    // Read in keys and values, keeping internal whitespace
    typedef string::size_type pos;
    const string& delim = cf.myDelimiter; // separator
    const string& comm = cf.myComment; // comment
    const string& sentry = cf.mySentry; // end of file sentry
    const pos skip = delim.length(); // length of separator

    string nextline = ""; // might need to read ahead to see where value ends

    while (is || nextline.length() > 0)
    {
        // Read an entire line at a time
        string line;
        if (nextline.length() > 0)
        {
            line = nextline; // we read ahead; use it now
            nextline = "";
        } else
        {
            std::getline(is, line);
        }

        // Ignore comments
        line = line.substr(0, line.find(comm));

        // Check for end of file sentry
        if (sentry != "" && line.find(sentry) != string::npos)
            return is;

        // Parse the line if it contains a delimiter
        pos delimPos = line.find(delim);
        if (delimPos < string::npos)
        {
            // Extract the key
            string key = line.substr(0, delimPos);
            line.replace(0, delimPos + skip, "");

            // See if value continues on the next line
            // Stop at blank line, next line with a key, end of stream,
            // or end of file sentry
            bool terminate = false;
            while (!terminate && is)
            {
                std::getline(is, nextline);
                terminate = true;

                string nlcopy = nextline;
                ConfigReader::trim(nlcopy);
                if (nlcopy == "")
                    continue;

                nextline = nextline.substr(0, nextline.find(comm));
                if (nextline.find(delim) != string::npos)
                    continue;
                if (sentry != "" && nextline.find(sentry) != string::npos)
                    continue;

                nlcopy = nextline;
                ConfigReader::trim(nlcopy);
                if (nlcopy != "")
                    line += "\n";
                line += nextline;
                terminate = false;
            }

            // Store key and value
            ConfigReader::trim(key);
            ConfigReader::trim(line);
            cf.myContents[key] = line; // overwrites if key is repeated
        }
    }

    return is;
}


