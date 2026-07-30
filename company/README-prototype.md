
# Real-Time Machine Learning-Based Threat Detection for ISP Network Traffic





## How to run the program

1. Log in to the VM and go to the app directory

```
cd app

```
2. Activate the virtual environment (for python libraries)

``` 
source venv/bin/activate 

```
3. Open the databases 

``` 
docker start redis-server && docker start offline-db

```
4. Run Suricata 

```
sudo suricata  -c /home/atoma/oisf/suricata/suriredis.yaml -i dummy --af-packet

```

5. Run tcpreplay

```
 cd pcaps && sudo tcpreplay -i dummy firewall_test_20250730_115700.pcap

```
6. Run Streamlit frontend (main - first page for the live predictions , connectiondb - second page for amodel training)

```
streamlit run nav.py 

```
7. Access the web page 

```
CTRL + MOUSE CLICK  on the last link 

```


# Redis

## How to access redis-cli 

In the terminal run the following command 
```
redis-cli -h 0.0.0.0 -p 6379

```
## How to clear the redis stream  
In the cli, run the following command 
```
xtrim suricata:packets MAXLEN 0
```


# PostgreSQL - database 

## How to open the database 

```
docker exec -it offline-db psql -U user -d offline-vector
```
# Suricata plugin 

In the VM, there are 2 plugins for Suricata in 2 different folders 

* ```pluginredis ``` folder - plugin used for retrive data from the interface and sending data to the data stream 

* ```my_plugin ``` folder - plugin made for creating the labelded dataset 


 The plugins are linked to a specific yaml file: 

 * ``` custom-out.so ```(my_plugin) : ``` suri2.yaml ```(oisf/suricata)
 * ``` custom-redis.so ``` (pluginredis) : ``` suriredis.yaml ```(oisf/suricata)

 ## How to update the plugin
 After modifying the code from the c file, run the following command 

 ``` 
 make clean 
 
 ``` 
 and then 

 ```
 make 

 ```
 

     

