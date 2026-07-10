#!/usr/bin/env python3
import socket
import sys
import argparse

"""
LeftHandController={"id": "LeftHandController","motionTowardObject": "","targetMotionMode": 2,"targetPoint": {"x": -0.5,"y": 1.1,"z": 0.0},"translateSpeed": 1.0,"translateTime": -1,"translateAngularSpeed": -1,"targetRotation": {"x":0.0,"y": 30.0,"z": 0.0},"rotateSpeed": 270,"rotateTime": -1,"keepTime": 0,"mode": 2,"gazeTracking": true,"priority": 0,"isBezierCurvePoint": false,"fingerData": []}

RightHandController={"id": "RightHandController","motionTowardObject": "","targetMotionMode": 2,"targetPoint": {"x": 0.2,"y": 0.9,"z": 0.0},"translateSpeed": 1.0,"translateTime": -1,"translateAngularSpeed": -1,"targetRotation": {"x":0.0,"y": 1.0,"z": 0.0},"rotateSpeed": 270,"rotateTime": -1,"keepTime": 0,"mode": 2,"gazeTracking": true,"priority": 0,"isBezierCurvePoint": false,"fingerData": []}
"""


class TCPClient:
    """
    A TCP client for sending messages to a server.
    This class can be imported and used in other files.
    """

    def __init__(self, server="localhost", port=None):
        """
        Initialize the TCP client.

        Args:
            server (str): Server address (default: localhost)
            port (int): Server port (required)
        """
        self.server_address = server
        self.server_port = port
        self.client_socket = None
        self.connected = False
        self._buffer = ""

    def connect(self):
        """
        Connect to the server.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        if self.server_port is None:
            raise ValueError("Port number is required")

        try:
            # Create a TCP socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to the server
            print(f"Connecting to {self.server_address}:{self.server_port}...")
            self.client_socket.connect((self.server_address, self.server_port))
            print(f"Connected to {self.server_address}:{self.server_port}")
            self.connected = True
            return True

        except ConnectionRefusedError:
            print(
                f"Connection to {self.server_address}:{self.server_port} was refused. Make sure the server is running."
            )
        except socket.gaierror:
            print(
                f"Address-related error connecting to {self.server_address}:{self.server_port}"
            )
        except socket.error as e:
            print(f"Socket error: {e}")

        return False

    def send_message(self, message):
        """
        Send a message to the server.

        Args:
            message (str): The message to send

        Returns:
            bool: True if message sent successfully, False otherwise
        """
        if not self.connected or self.client_socket is None:
            print("Not connected to any server. Call connect() first.")
            return False

        try:
            # Add newline if not present
            if not message.endswith("\n"):
                message += "\n"

            # Send the message
            self.client_socket.sendall(message.encode("utf-8"))
            return True
        except socket.error as e:
            print(f"Error sending message: {e}")
            self.connected = False
            return False

    def receive_data(self, buffer_size=1024):
        """
        Receive data from the server.

        Args:
            buffer_size (int): The buffer size for receiving data

        Returns:
            str: The received data or None if there was an error
        """
        if not self.connected or self.client_socket is None:
            print("Not connected to any server. Call connect() first.")
            return None

        try:
            while "\n" not in self._buffer:
                data = self.client_socket.recv(buffer_size)
                if not data:
                    return None
                self._buffer += data.decode("utf-8")
            newline_idx = self._buffer.index("\n")
            message = self._buffer[:newline_idx]
            self._buffer = self._buffer[newline_idx + 1 :]
            return message
        except socket.error as e:
            print(f"Error receiving data: {e}")
            return None

    def close(self):
        """Close the connection to the server."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.connected = False
            print("Connection closed")

    def interactive_mode(self):
        """Run an interactive session where the user can type messages to send."""
        if not self.connected:
            success = self.connect()
            if not success:
                return

        try:
            print("Type your message and press Enter to send. Enter 'exit' to quit.")
            # Main loop for sending messages
            while True:
                message = input("> ")

                # Check if user wants to exit
                if message.lower() == "exit":
                    break

                # Send the message
                self.send_message(message)

        except KeyboardInterrupt:
            print("\nInteractive mode terminated by user")
        finally:
            self.close()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="TCP Client to send messages to a server"
    )
    parser.add_argument(
        "-s",
        "--server",
        default="localhost",
        help="Server address (default: localhost)",
    )
    parser.add_argument("-p", "--port", type=int, required=True, help="Server port")
    args = parser.parse_args()

    # Create a client instance
    client = TCPClient(server=args.server, port=args.port)

    # Run in interactive mode
    client.interactive_mode()


if __name__ == "__main__":
    main()
