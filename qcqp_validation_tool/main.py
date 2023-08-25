import rclpy
from rclpy.node import Node
import pandas as pd
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import networkx as nx
import numpy as np
import pandas as pd
from gurobipy import GRB, Model
from consts import COMPONENTS, LEVEL_ORDER

TOPIC_NAME = 'df_publisher'

class ValidationNode(Node):

    def __init__(self):
        super().__init__('validation_node')
        self.publisher = self.create_publisher(Float64MultiArray, 'validation', 10)
        self.subscription = self.create_subscription(
            Float64MultiArray,
            TOPIC_NAME,
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
        self.publish_dataframe(msg.data)

    def publish_dataframe(self,data):
        array_2d = np.array(data).reshape(-1, 23)
        df = pd.DataFrame(array_2d, columns=["id", "x", "y", "z"] + COMPONENTS)
        cords = self.validate_detections(df=df)
        msg = Float64MultiArray()
        msg.data = cords
        self.publisher.publish(msg)


    def calculate_distance(self, point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


    def find_closest_node(self, G1, G2):
        """Finds the closest G1 node for each node in G2"""
        closest_nodes = {}
        for g2_node in G2.nodes(data=True):
            g2_pos = g2_node[1]["pos"]
            min_dist = float("inf")
            min_node = None
            for g1_node in G1.nodes(data=True):
                g1_pos = g1_node[1]["pos"]
                dist = self.calculate_distance(g1_pos, g2_pos)
                if dist < min_dist:
                    min_dist = dist
                    min_node = g1_node[0]
            closest_nodes[g2_node[0]] = min_node
        return closest_nodes

    def validate_detections(self, df):
        detected = df.copy()
        static = pd.read_csv("qcqp_validation_tool/nodes_info.csv")
        quadratic_model = Model("quadratic")
        variables = {}
        for name in LEVEL_ORDER:
            variables[f"{name}_x"] = quadratic_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_x"
            )
            variables[f"{name}_y"] = quadratic_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_y"
            )
            variables[f"{name}_z"] = quadratic_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_z"
            )

        obj_fn = 0
        for row in detected.iterrows():
            detected_x = row[1]["x"]
            detected_y = row[1]["y"]
            detected_z = row[1]["z"]
            for component in COMPONENTS:
                confidence = row[1][component]
                term = confidence * (
                    (variables[f"{component}_x"] - detected_x) ** 2
                    + (variables[f"{component}_y"] - detected_y) ** 2
                    + (variables[f"{component}_z"] - detected_z) ** 2
                )
                obj_fn += term


        for name in LEVEL_ORDER:
            node_info = static[static["name"] == name]
            parent = node_info["parent"].values[0]
            distance_to_parent = node_info["distance_to_parent"].values[0]
            if name == "root_cross_left":
                quadratic_model.addQConstr(variables[f"{name}_x"] == -13)
                quadratic_model.addQConstr(variables[f"{name}_y"] == -20)
                quadratic_model.addQConstr(variables[f"{name}_z"] == 0)

            elif name == "root_cross_right":
                quadratic_model.addQConstr(variables[f"{name}_x"] == 10)
                quadratic_model.addQConstr(variables[f"{name}_y"] == -1)
                quadratic_model.addQConstr(variables[f"{name}_z"] == 0)
            else:
                quadratic_model.addQConstr(
                    (
                        (variables[f"{name}_x"] - variables[f"{parent}_x"]) ** 2
                        + (variables[f"{name}_y"] - variables[f"{parent}_y"]) ** 2
                        + (variables[f"{name}_z"] - variables[f"{parent}_z"]) ** 2
                    )
                    <= distance_to_parent**2
                )


        quadratic_model.setObjective(obj_fn, GRB.MINIMIZE)

        quadratic_model.optimize()

        G1 = nx.Graph()
        for i in range(len(detected)):
            G1.add_node(
                str(detected["id"][i]),
                pos=(detected["x"][i], detected["z"][i], detected["y"][i]),
            )


        G2 = nx.Graph()
        for name in LEVEL_ORDER:
            if variables[f"{name}_x"].x != 250 and variables[f"{name}_x"].x != -250:
                G2.add_node(
                    name,
                    pos=(
                        variables[f"{name}_x"].x,
                        variables[f"{name}_z"].x,
                        variables[f"{name}_y"].x,
                    ),
                )

        closest_nodes = self.find_closest_node(G2, G1)
        df_closest_nodes = pd.DataFrame(
            closest_nodes.items(), columns=["G1_Node", "Closest_G2_Node"]
        )
        df_closest_nodes = df_closest_nodes.set_index("G1_Node")
        df_closest_nodes.sort_values(by=["Closest_G2_Node"], inplace=True)
        detected['valid'] = detected['id'].astype(str).map(df_closest_nodes['Closest_G2_Node'])



        final_model = Model("quadratic")

        f_vars = {}
        for name in LEVEL_ORDER:
            f_vars[f"{name}_x"] = final_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_x"
            )
            f_vars[f"{name}_y"] = final_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_y"
            )
            f_vars[f"{name}_z"] = final_model.addVar(
                vtype=GRB.CONTINUOUS, lb=-250, ub=250, name=f"{name}_z"
            )



        obj_fn = 0
        for row in detected.iterrows():
            detected_x = row[1]["x"]
            detected_y = row[1]["y"]
            detected_z = row[1]["z"]
            component = row[1]["valid"]
            term = (
                (f_vars[f"{component}_x"] - detected_x) ** 2
                + (f_vars[f"{component}_y"] - detected_y) ** 2
                + (f_vars[f"{component}_z"] - detected_z) ** 2
            )
            obj_fn += term

        for name in LEVEL_ORDER:
            node_info = static[static["name"] == name]
            parent = node_info["parent"].values[0]
            distance_to_parent = node_info["distance_to_parent"].values[0]
            if name == "root_cross_left":
                final_model.addQConstr(f_vars[f"{name}_x"] == -13)
                final_model.addQConstr(f_vars[f"{name}_y"] == -20)
                final_model.addQConstr(f_vars[f"{name}_z"] == 0)

            elif name == "root_cross_right":
                final_model.addQConstr(f_vars[f"{name}_x"] == 10)
                final_model.addQConstr(f_vars[f"{name}_y"] == -1)
                final_model.addQConstr(f_vars[f"{name}_z"] == 0)
            else:
                final_model.addQConstr(
                    (
                        (f_vars[f"{name}_x"] - f_vars[f"{parent}_x"]) ** 2
                        + (f_vars[f"{name}_y"] - f_vars[f"{parent}_y"]) ** 2
                        + (f_vars[f"{name}_z"] - f_vars[f"{parent}_z"]) ** 2
                    )
                    <= distance_to_parent**2
                )

        final_model.setObjective(obj_fn, GRB.MINIMIZE)
        final_model.optimize()


        final_cords = []
        for name in LEVEL_ORDER:
            final_cords += [round(f_vars[f"{name}_x"].x, 2),round(f_vars[f"{name}_y"].x, 2),round(f_vars[f"{name}_z"].x, 2)]
        return final_cords





def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = ValidationNode()

    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()