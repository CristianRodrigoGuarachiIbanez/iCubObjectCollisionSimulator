# Short intro to the iCub control system
The iCub is not controlled low-level by communicating directly with the hardware. This would be very complicated and the software would be very robot specific. Therefore a robotic middleware is used to wrap the hardware control. In the case of the iCub it is [YARP](http://www.yarp.it/index.html). [Here](http://www.yarp.it/what_is_yarp.html) it is explained more detailed. For a deeper dive into the yarp ecosphere see the [tutorials](http://www.yarp.it/tutorials.html)

The YARP-control is based on devices with a [port-based communication system](http://www.yarp.it/note_ports.html). This allows the distribution of program modules on different machines. Enabling the possibility to use different OS in the same control project and the computational power of clusters. 

There are devices for all the different features. The plan in this documentation is, to give a short overview for the most important ones. The first section is the motor control system. The visual and tactile sensing will follow soon. 

## Programming with the iCub 
To facilitate the programming of software using the iCub, there exists two packages, which wrap parts of the YARP-based control.

The first one is a set of [Python "Libraries"](https://ai.informatik.tu-chemnitz.de/gogs/iCub_TUC/iCub_Python_Lib.git), which contains methods for motor control (at the moment only Position and Velocity) and visual perception. Beside there are classes for manipulating the simulation worlds in the iCub- or gazebo-simulator, using the YARP-worldinterface. Which inludes object and 3D-model manipulation.

The second package is designed as an interface between the iCub robot and ANNarchy. This consists of different modules for the sensor and actuator control.  
I consists of four modules:
- JointWriter:
    - This module handles the joint motion. At this point for the joint control only the Position control is included. The control is extended with the possiblity to directly feed population-coded joint angles to the robot.
- JointReader:
    - This module wraps the reading of the joint angles from the encoders.
- VisualReader:
    - In this module the handling of the camera images is done. It allows monocular or binocular mode and returns the images in a grayscale 1D-vector.
- SkinReader
    - Here the tactile sensing of the robot is wrapped. At the moment only for the robot arms. Beside the sensor values, the sensor positions are accessible.

This interface is placed in https://ai.informatik.tu-chemnitz.de/gogs/iCub_TUC/Interface_ANNarchy_iCub.git. For further information see the interface [ReadMe](https://ai.informatik.tu-chemnitz.de/gogs/iCub_TUC/Interface_ANNarchy_iCub/src/master/ReadMe.md).
