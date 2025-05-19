启动carla服务器
./CarlaUE4.sh -quality-level=Epic
./CarlaUE4.sh -RenderOffScreen

生成交通要素
python ~/carla/PythonAPI/examples/generate_traffic.py --safe -n 50 -w 0 --asynch --port 2000

启动采集
python start_sampling.py

数据处理
- 遮挡程度 python data_process.py
