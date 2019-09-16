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
