import rospy
import tf

def broadcast_world_frame():
    rospy.init_node('world_frame_broadcaster', anonymous=True)
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        br.sendTransform(
            (0.0, 0.0, 0.0),  # Translation: origin
            (0.0, 0.0, 0.0, 1.0),  # Rotation: identity quaternion
            rospy.Time.now(),
            "world",  # Child frame
            "base_link"  # Parent frame
        )
        rate.sleep()

if __name__ == '__main__':
    try:
        broadcast_world_frame()
    except rospy.ROSInterruptException:
        pass
