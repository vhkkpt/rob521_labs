#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

def sync_rostime():
    while rospy.get_rostime().to_sec() == 0:
        pass

def publisher_node():
    cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size = 5)
    rate = rospy.Rate(200)
    ts = rospy.get_rostime()

    while (rospy.get_rostime() - ts).to_sec() < 10:
        twist = Twist()
        twist.linear.x = 0.1
        twist.angular.z = 0
        cmd_pub.publish(twist)
        rate.sleep()

    ts = rospy.get_rostime()

    while (rospy.get_rostime() - ts).to_sec() < math.pi * 2 / 0.5:
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0.5
        cmd_pub.publish(twist)
        rate.sleep()

    twist = Twist()

    while not rospy.is_shutdown():
        cmd_pub.publish(twist)
        rate.sleep()

def main():
    try:
        rospy.init_node('motor')
        sync_rostime()
        publisher_node()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
