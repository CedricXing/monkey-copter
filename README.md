# Monkey-copter

### Set up ArduPilot development environment
#### Get git

[Git](https://git-scm.com/) is a free and open source distributed version control system that is used to manage ArduPilot codebase. Git is available on all major OS platforms, and a variety of tools exist to make it easier to get started.

For Linux/Ubuntu users can install with apt
```
sudo apt-get update
sudo apt-get install git
sudo apt-get install gitk git-gui 
```
For other systems, please download and install following this [page ](https://git-scm.com/).

#### Clone ArduPilot repository

Firstly, make a new directory, for example dir, as the working environment. 
```
mkdir dir
cd dir
```
Then, use git to clone the ArduPilot repository and update the dependency submodule of ArduPilot.
```
git clone https://github.com/CedricXing/arduPilot.git
cd arduPilot
git submodule update --init --recursive
```
#### Install some required packages
If you are on a debian based system (such as Ubuntu or Mint), the ArduPilot community provides a script that will do it for you. From ArduPilot directory, run
```
Tools/environment_install/install-prereqs-ubuntu.sh -y
```
Then, reload the path (or log-out and log-in to make permanent)
```
. ~/.profile
```
Now, you should be able build with waf. Try
```
make sitl -j4
```
**IMPORTANT**: go to ~/.local/lib/python2.7/site-packages/MAVProxy/modules, and edit mavproxy_link.py to remove the line with MAV_TYPE_DODECAROTOR.

Now, you can run the SITL simulator. For example, for the multicopter code, go to the ArduCopter directory and start simulating using **sim_vehicle.py**.
```
cd ArduCopter
sim_vehicle.py -w 
sim_vehicle.py --console --map
```
The third command above can be used to launch a map.

### Install dronekit-python
[DroneKit-Python](https://dronekit.netlify.com/) contains the python language implementation of DroneKit.

The API allows developers to create Python apps that communicate with vehicles over MAVLink. It provides programmatic access to a connected vehicle's telemetry, state and parameter information, and enables both mission management and direct control over vehicle movement and operations.

To install dronekit-python, run
```
sudo pip install dronekit
```
Probably you need to install the following packages first:
```
sudo apt-get install python-pip python-dev

sudo apt-get install python-pip python-dev python-numpy python-opencv python-serial python-pyparsing python-wxgtk2.8
```
Then, install the dronekit package for SITL
```
sudo pip install dronekit-sitl -UI
```
**Note**: dronekit only supports Python2.7.

### Set up monkey-copter
Monkey-copter is an automatic human-user mimicking program that is created to run the simulations automatically. You can download it by git using the following command
```
cd ArduPilot
git clone https://github.com/CedricXing/monkey-copter.git
```
Go to the monkey-copter and revise the configuration file `config.ini`. Then try
```
nohup python2.7 script.py &
```
to run the monkey program as a background process.
