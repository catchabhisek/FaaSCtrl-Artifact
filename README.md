# FaaSCtrl
FaaSCtrl is a intra-node resource manager tailored for serverless platforms. The objective of the manager is to ensure that the components of the response latency (mean, median, tail and std. dev.) is within a prespecified limit for an LS application, and for BE applications, these components are minimized on a best-effort basis. Instead of using standard approaches based on optimization theory, it uses a much faster reinforcement learning (RL) based approach to tune the knobs that govern process scheduling in Linux, namely the real-time priority and the assigned number of cores.

## Getting Started
FaaSCtrl currently runs along with Apache OpenWhisk (an opensource serverless platform). You can install Apache OpenWhisk from the following link: https://github.com/apache/openwhisk.  Note that we deploy Apache OpenWhisk using Apache CouchDB. Now let us discuss its integration with FaaSCtrl.

### Code Structure
* `resource-manager` contains the RL framework of FaaSCtrl.
* `database` contains the python script to collect latency statistics of applications.
* `state-monitor` contains the bash scripts that collects an application's hardware performance events and CPU usage events during function execution. Furthermore, it is also responsible for setting an application's real-time priority and the assigned number of cores.


### FaaSCtrl Deployment
1. Add the FAAS ROOT directory and the list of *LS* applications and other necessary information to the `faasched.py`, `adjust_priority.sh`, and `poll_state.sh` scripts.
2. Start the intra-node resource manager by executing the following command:
    ```
    python3 resource-manager/faasched.py
    ```

**Note**: We can also use it with other serverless platforms with minimal changes.

## License
FaaSCtrl is licensed under the Apache v2 License.