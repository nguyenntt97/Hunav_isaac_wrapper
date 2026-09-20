#!/usr/bin/env python3
"""
robots/ros_publishers.py

Publishes a robot's odometry, TF and joint states from Python.

Deliberately not an OmniGraph. The simulation node is already an
``rclpy.node.Node`` handling /cmd_vel in Python, and running the ROS 2 bridge
extension's publishers alongside a live rclpy context is what the go2_omniverse
fork documents as tripping a duplicated ``rcl_interfaces`` ParameterEvent
assertion (see its run_sim.sh). One publishing mechanism per process.

Only robots whose USD does not already publish these get a publisher --
carter_ROS carries its own OmniGraph and would otherwise emit /odom twice.
"""

import math

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


def _yaw_from_quaternion(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RobotStatePublisher:
    """/odom, /tf (odom -> base_link) and /joint_states for one robot driver."""

    def __init__(self, node, driver, odom_frame="odom", base_frame="base_link"):
        self.node = node
        self.driver = driver
        self.odom_frame = odom_frame
        self.base_frame = base_frame

        self.odom_pub = node.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = node.create_publisher(TFMessage, "/tf", 10)
        self.joint_pub = node.create_publisher(JointState, "/joint_states", 10)

    def publish(self):
        """Publish one sample. Safe to call before physics has started."""
        try:
            position, orientation = self.driver.get_world_pose()
            linear = self.driver.get_linear_velocity()
            angular = self.driver.get_angular_velocity()
        except Exception:
            # The articulation is not live yet; there is nothing to say.
            return

        stamp = self.node.get_clock().now().to_msg()
        w, x, y, z = (float(v) for v in np.asarray(orientation).reshape(-1)[:4])
        px, py, pz = (float(v) for v in np.asarray(position).reshape(-1)[:3])

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = px
        transform.transform.translation.y = py
        transform.transform.translation.z = pz
        transform.transform.rotation.w = w
        transform.transform.rotation.x = x
        transform.transform.rotation.y = y
        transform.transform.rotation.z = z
        self.tf_pub.publish(TFMessage(transforms=[transform]))

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.w = w
        odom.pose.pose.orientation.x = x
        odom.pose.pose.orientation.y = y
        odom.pose.pose.orientation.z = z

        # REP-105: the twist in an Odometry message is expressed in the child
        # frame, but Isaac reports world-frame velocities, so rotate by -yaw.
        linear = np.asarray(linear).reshape(-1)
        angular = np.asarray(angular).reshape(-1)
        yaw = _yaw_from_quaternion(w, x, y, z)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx, vy = float(linear[0]), float(linear[1])
        odom.twist.twist.linear.x = cos_yaw * vx + sin_yaw * vy
        odom.twist.twist.linear.y = -sin_yaw * vx + cos_yaw * vy
        odom.twist.twist.linear.z = float(linear[2])
        odom.twist.twist.angular.x = float(angular[0])
        odom.twist.twist.angular.y = float(angular[1])
        odom.twist.twist.angular.z = float(angular[2])
        self.odom_pub.publish(odom)

        joints = self.driver.joint_state()
        if joints is not None:
            names, positions = joints
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = list(names)
            msg.position = [float(v) for v in np.asarray(positions).reshape(-1)]
            self.joint_pub.publish(msg)
