import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

TOPIC_NAME = 'df_publisher' 

class DataFramePublisher(Node):
    def __init__(self):
        super().__init__('dataframe_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, TOPIC_NAME, 10)
        self.publish_dataframe()

    def publish_dataframe(self):
        df = pd.read_csv("/packages/src/qcqp_validation_tool/qcqp_validation_tool/sample_data.csv")
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='rows', size=df.shape[0], stride=df.shape[0]*df.shape[1]),
            MultiArrayDimension(label='columns', size=df.shape[1], stride=df.shape[1])
        ]
        msg.layout.data_offset = 0

        msg.data = df.values.flatten(order='F').tolist()

        self.publisher.publish(msg)

rclpy.init()
node = DataFramePublisher()
rclpy.spin(node)
