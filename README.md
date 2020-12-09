# ArduPilot Testbed Setup

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
#### Clone monkey-copter repository
Monkey-copter is an automatic human-user mimicking program that is created to run the simulations automatically. It can be downloaded by git using the following command
```
cd ArduPilot
git clone https://github.com/CedricXing/monkey-copter.git
```
#### Run the simulations
We prepare the script and configuration file to run the simulations automatically. Firstly, enter the monkey-copter and revise the configuration file `config.ini`. The configuration parameters including

* **root_dir** : the root directory of ArduPilot
* **real_life** : 'True' for real-life bug subjects and 'False' for artificial bug subjects
* **mutiple_bugs** : 'True' for mutiple bugs (5 bugs by default) in each subject and 'False' for only one bug in each subject
* **start** : the starting serial number of one subject
* **end** : the ending serial number of one subject
* **rounds** : the number of bug subjects

After finish setting configurations, try
```
nohup python2.7 script.py &
```
to run the monkey program as a background process. The simulation results will be exported to `experiment/output/` and the corresponding running configuration files will be exported to `experiment` for the future use in our proposed **Autoregression** labeling method. 

# IP+CV Testbed Setup (based on Mac OS)

### Anaconda Setup

#### Install Anaconda
[Anaconda](https://www.anaconda.com/) is a popular and free platform where packages, notebooks, projects and environments are shared. To install Anaconda, please follow the [official website](https://docs.anaconda.com/anaconda/install/). 

After installing the anaconda, we need to change the `.zshrc` to add the path of anaconda. If you are using Zsh (Bash can be added similarly), try
```
export PATH=/Users/yourname/anaconda/bin:$PATH
```
Remember to modify the 'yourname' to your own username. Then, restart the terminal or `source ~/.zshrc` to make the modification come to effect.

#### Create local conda environment
Now, we create a local conda environment for IP+CV testbed. Since we use opencv3 and python2.7, we create a python environment called `opencv3_python27` by
```
conda create --name opencv3_python27 python=2.7
```
To activate this environment, use the command `conda activate opencv3_python27`. You can refer to more information about how to create local environments, activate the environments and deactivate them in the offical site.

### OpenCV Setup
#### Install OpenCV3 and Python2.7
The detailed installation tutorial of OpenCV3 and Python2.7 on Mac OS can be found by searching `macOS: Install Opencv 3 and Python 2.7`. Follow the Step 5, 6, 7, 8. Specifically, in Step 6, download `opencv` and `opencv_contrib` using the following commmands.
```
git clone https://github.com/CedricXing/lib_opencv
```

In Step 7, the following configuration for **Cmake** is used on our own Mac OS machine:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/Documents/ipcv/opencv_contrib/modules \
    -D PYTHON2_LIBRARY=/Users/yourname/anaconda/envs/opencv3_python27/lib/python2.7/config/libpython2.7.a \
    -D PYTHON2_INCLUDE_DIR=/Users/yourname/anaconda/envs/opencv3_python27/include/python2.7/ \
    -D PYTHON2_NUMPY_INCLUDE_DIRS=/Users/yourname/anaconda/envs/opencv3_python27/lib/python2.7/site-packages/numpy/core/include \
    -D PYTHON2_EXECUTABLE=/Users/yourname/anaconda/envs/opencv3_python27/bin/python2.7 \
    -D BUILD_opencv_python2=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=ON ..
```
Please modify the configuraton to your own path settings.

#### Link OpenCV to the conda environment
After running `make install`, you should now see a file named `cv2.so` in `/usr/local/lib/python2.7/site-packages/cv2/python-2.7/cv2.so`. You need to sim-link this file to your conda virtual environment. First, enter the directory of your conda virtual environment by
```
cd ~/anaconda/envs/opencv3_python27/lib/python2.7/site-packages
```
Then, use the sim-link command
```
ln -s /usr/local/lib/python2.7/site-packages/cv2/python-2.7/cv2.so cv2.so
```

#### Recompile Opencv3
In the previous step, you have already compiled the native opencv3 and link it to your conda virtual environment. If you insert bugs into opencv3 later, you can use the following commands to recompile and reinstall opencv3
```
make -j8
make install
```

#### Install Other Python Libraries
We also need to install `numpy`, `Pillow` and `SimpleWebSocketServer`. Try
```
pip install numpy Pillow
pip install git+https://github.com/dpallot/simple-websocket-server.git
```
### Build IP Plant
#### Install Unity3D
Download the version Unity 5.6.1f from [Unity](https://unity3d.com/). Since this tutorial is based on Mac OS, choose the Mac OS version. After installing Unity3D, download the IP Plant project from [here](https://surefire.comp.polyu.edu.hk/research/cps/software_engineering/). You need to revise the following file paths in `LQR_test.cs` to use the application.
| name | meaning |
|---- | ---- |
|file_path | the directory will output physical trajectories | 
|minor_file| the counter that records the current physical trajectory id|
|ip_stm_minor| the counter of that records the current subject id |

Modify them to the correct location of your own computer. Then, you can try building the IP Plant in Unity3D.

#### Run the Software Fault Localization(SFL) tools
We have implemented 6 SFL tools including `Tarantula`, `Crosstab`, `BPNN`, `DStar`, `Ochiai`, `Ochiai2`. Specifically, `BPNN` is based on neural network and we implement it by [Pytorch](https://pytorch.org/) framework. So we need to install `pytorch` first. We recommend [Anaconda](https://www.anaconda.com/) to install the relative python packages. Go to the [Anaconda-download page](https://www.anaconda.com/distribution/) to download the `Anaconda` installation package and then install it. After that, try
```
conda install pytorch torchvision cpuonly -c pytorch
```
to install the cpuonly-version `pytorch`.



