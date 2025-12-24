# ROS-Mani小车执行任务
任务说明：
图1场景中一共有5个位置，机器人需完成下列任务：
1）	从①自动巡航到②；
2）	识别②前方的仙女和强盗，把强盗击倒；
3）	自动巡航到③，抓取一个立方体，到④放到相应的颜色框
4）	自动巡航到⑤，识别前方的数字1或2，根据数字大小，机器人旋转相应圈数
5）	返回①

<img width="478" height="594" alt="image" src="https://github.com/user-attachments/assets/6b0f2f78-f888-4bbc-84d0-46a83d1fe50d" />

使用方法：
首先在虚拟机上装好ubuntu系统，在系统中安装ros，然后克隆mani功能包。在以上工作的基础上将mobile_pkg功能包粘贴进catkin_ws的src文件夹中，复制剩下两个文件夹将wpb_mani中的两个同名文件夹替换掉，将mobile_pkg文件夹中的waypoints文件粘贴至Home目录下。

完成以上工作后在catkin_ws文件夹中打开ubuntu的命令端，编译成功后输入roslaunch mobile_pkg kh6启动仿真地图环境，再打开一个命令端输入rosrun mobile_pkg cruise_node启动小车程序

该项目实现了小车巡航、机械臂抓取放置物品、目标识别等功能
