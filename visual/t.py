#!/usr/bin/env  python3.6
import sys,os
import datetime
import urllib.request
import linecache
import time
import getpass
from subprocess import Popen, PIPE
# Tested on Mac , manually add write privilege for /etc/hosts is required.
# Latest update : 10 Apr , 2017

url = 'https://coding.net/u/scaffrey/p/hosts/git/raw/master/hosts'
today = datetime.date.today()
fName = 'hosts' + str(today) #the downloaded hosts file from internet
origHosts = '/etc/hosts'
startLine = 0 #delete and appand data from the startLine of system hosts file
date_line = startLine + 3 # # Last updated: 2017-03-14 , specify the date line in your hosts file
warn = '''

#############################################################################


This script supported on Mac only . Trying to run on windows platform is about to fail.


#############################################################################








'''

# run commands with sudo
def runasSudo(command, passwd):
    p = Popen('sudo -S %s' % (command), shell=True, stdin=PIPE, stderr=PIPE, stdout=PIPE, universal_newlines=True)
    p.communicate(passwd + '\n')[1]

# get system password
def getPass():
    passwd = getpass.getpass(prompt='Please input your system password, press enter to continue: ')
    return passwd
def lineNum():
    with open(origHosts, 'r') as f:
        lines = len(f.readlines())
    return lines
# check if current user is able to write data to origHosts, it is a readonly file by default
def isWritable():
    return os.access(origHosts, os.W_OK)

def grantPriv():
    # os.system('echo %s | sudo -S chmod a+w %s' % (passwd, origHosts))
    return 'chmod a+w %s' % origHosts
# get date from online hosts file , it is downloaded and stored in the current directory
def getDate(file_name, linenum):
    strDate = linecache.getline(file_name, linenum).split(':')[1].strip()
    year_s, mon_s, day_s = strDate.split('-')
    return datetime.date(int(year_s), int(mon_s), int(day_s))

# delete the data from startLines which defined , default is 0 means delete all data from origHosts
def delData():
    reserve = []
    try:
        with open(origHosts, 'r') as f:
            for line in f.readlines()[:startLine]:
                reserve.append(line)
        with open(origHosts, 'w') as g:
            for i in range(0, len(reserve)):
                g.write(reserve[i])

    except Exception as e:
        print(e)

# write the latest content to system hosts file
def updateHosts(passwd):
    print('Updating hosts file...')
    try:
        delData()
        with open(origHosts, 'a') as f:
            with  open(fName, 'r') as g:
                f.write(g.read())
        print('%s has been updated successfully' % origHosts)
        flushDNS(passwd)
    except Exception as e:
        print(e)
    finally:
        delHosts()

# flush system dns cache
def flushDNS(passwd):
    commandList = ["dscacheutil -flushcache", "killall -HUP mDNSResponder", "say DNS cache flushed"]
    for command in commandList:
        print('%s ...' % command)
        runasSudo(command, passwd)

# check the hosts update from internet.
def date_comp(passwd):
    online_version = getDate(fName, 3)
    local_version = getDate(origHosts, date_line)
    print('The latest online version is %s' % (str(online_version)))
    print('The local version is %s' % (str(local_version)))
    if online_version > local_version and lineNum() > 1000 :
        updateHosts(passwd)
    else:
        print('You are using the lastest hosts , enjoy it !')

# It will remove the hosts file downloaded from internet by default , reset flag value to '0' if you want to reserve it : flag=0
def delHosts(flag=1):
    if flag == 1:
        os.remove(fName)
    else:
        pass

# download the latest version of hosts file from internet and saved at current directory
def host_download():
    if not os.path.exists(fName):
        sys.stdout.write('\rFetching latest hosts file from internet...\n')
        urllib.request.urlretrieve(url, fName)
        sys.stdout.write("\rDownload complete, saved as %s " % (fName) + '\n\n')
        sys.stdout.flush()


def main():
    print(warn)
    passwd = getPass()
    if not isWritable():
        # print('granting write privilege with sudo')
        runasSudo(grantPriv(), passwd)
    else:
        print('%s is writable' % (origHosts))

    host_download()
    date_comp(passwd)

if __name__ == '__main__':
    main()
