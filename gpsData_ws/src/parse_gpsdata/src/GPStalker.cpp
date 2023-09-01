#include <string>
#include <array>
#include <vector>
#include <sstream>

#include <ros/ros.h>                           // 包含ROS的头文件
#include <ros/time.h>
#include <ros/duration.h>
#include <boost/asio.hpp>                  //包含boost库函数
#include <boost/bind.hpp>
#include "std_msgs/String.h"              //ros定义的String数据类型
#include "geometry_msgs/Vector3.h"

#include "parse_gpsdata/GPSDATA.h"

using namespace std;
using namespace boost::asio;           //定义一个命名空间，用于后面的读写操作
using namespace parse_gpsdata;

class GFPFPD_INFO {
public:
    GFPFPD_INFO() = default;
    void GPFDP_Analyse(const vector<string>&);
    void GPFDP_publiser(ros::Publisher&);
private:
	size_t GPSweek;             //从1980-1-6至当前的星期数
	double GPSTime;    //从周日00:00开始到当前的秒数
	float heading;              //偏航角
	float pitch;				//俯仰角
	float Roll;					//滚转角
	double Latitude;			//纬度 deg
	double Longitude;			//经度 deg
	float Altitude;				//高度 m
	float Ve;					//东 速度 m/s
	float Vn;					//北 速度 m/s
	float Vu;					//天 速度 m/s
	float Baseline;				//基线长度
	int NSV1;					//天线1卫星数
	int NSV2;					//天线2卫星数
	int status;		//系统状态
    
};

GFPFPD_INFO gpfpd_data;
int i = 0;

int main(int argc, char** argv) {

    clock_t begin_time;
    clock_t end_time;
    char buf[1];
    vector<string> datas = vector<string>(15, "");

    setlocale(LC_ALL, ""); //防止中文乱码

    ros::init(argc, argv, "gpsdata");       //初始化节点
    ros::NodeHandle nh;                     //初始化句柄

    ros::Publisher pub = nh.advertise<const parse_gpsdata::GPSDATA>("GPFPD", 10); //话题名为GPFPD,等待队列长度1000


	/////////////////////////////////////////////////////////////////
    io_service iosev;
    serial_port sp(iosev, "/dev/ttyUSB0");         //定义传输的串口
    sp.set_option(serial_port::baud_rate(115200));   
    sp.set_option(serial_port::flow_control());
    sp.set_option(serial_port::parity());
    sp.set_option(serial_port::stop_bits());
    sp.set_option(serial_port::character_size(8)); //八位0x**
	/////////////////////////////////////////////////////////////////

    ros::Rate rate(1000);


	while(ros::ok()) {
        //判断是不是$GPGPD协议
        string isnotGPFPD;  //判断是不是$GPGPD协议的变量
        // begin_time = clock();   //开始计时
        while(isnotGPFPD != "\n$GPFPD") {
            isnotGPFPD = string();
            read(sp, boost::asio::buffer(buf));
            while(*buf != 0x2c) {
                if(*buf == '\n') isnotGPFPD = string();
                isnotGPFPD += *buf;
                read(sp, boost::asio::buffer(buf));
            }
            // end_time = clock();
            // double duration = double((end_time - begin_time) / CLOCKS_PER_SEC);
            // if(duration > 5) { ROS_INFO("超过5s没有接收到$GPFPD协议数据了。"); return -1; }
        }
        // cout << isnotGPFPD << endl; 
        //获取去掉协议头之后的数据串
        auto iter = datas.begin();
        read(sp, boost::asio::buffer(buf)); 
        while(*buf != '\n') { //判断整个数据串是不是结束了

            if(*buf != 0x2c)  { *iter += *buf; } //判断是不是逗号
            else  if(iter != datas.end()) { ++iter; } 

            read(sp, boost::asio::buffer(buf)); 
        }

        //处理datas中数据
        gpfpd_data.GPFDP_Analyse(datas);
        //发布数据
        gpfpd_data.GPFDP_publiser(pub);

        rate.sleep();

        ros::spinOnce();

        vector<string>(15, "").swap(datas);
    }

    iosev.run(); 
    return 0;
}

void GFPFPD_INFO::GPFDP_Analyse(const vector<string>& datas) {
    
    int i = 0;
    for(auto& str: datas) {
        istringstream in(str);
        switch(i++) {
            case 0: in >> this->GPSweek; break;
            case 1: in >> this->GPSTime; break;
            case 2: in >> this->heading; break;
            case 3: in >> this->pitch; break;
            case 4: in >> this->Roll; break;
            case 5: in >> this->Latitude; break;
            case 6: in >> this->Longitude; break;
            case 7: in >> this->Altitude; break;
            case 8: in >> this->Ve; break;
            case 9: in >> this->Vn; break;
            case 10: in >> this->Vu; break;
            case 11: in >> this->Baseline; break;
            case 12: in >> this->NSV1; break;
            case 13: in >> this->NSV2; break;
            case 14: in >> this->status; break;
            default:break;
        }
    }

}

void GFPFPD_INFO::GPFDP_publiser(ros::Publisher& pub) {
    parse_gpsdata::GPSDATA gpsdata;
    gpsdata.Altitude = this->Altitude;
    gpsdata.Baseline = this->Baseline;
    gpsdata.GPSTime = this->GPSTime;
    gpsdata.GPSweek = this->GPSweek;
    gpsdata.Heading = this->heading;
    gpsdata.Latitude = this->Latitude;
    gpsdata.Longitude = this->Longitude;
    gpsdata.Nsv1 = this->NSV1;
    gpsdata.Nsv2 = this->NSV2;
    gpsdata.Pitch = this->pitch;
    gpsdata.Roll = this->Roll;
    gpsdata.Status = this->status;
    gpsdata.Ve = this->Ve;
    gpsdata.Vn = this->Vn;
    gpsdata.Vu = this->Vu;
    pub.publish(gpsdata);
    ROS_INFO("输出的数据为：%ld, %lf, %lf, %f, %lf, %lf, %f, %lf, %lf, %f, %lf, %d, %d, %d", 
    this->GPSweek, this->GPSTime, this->heading, this->pitch, this->Roll, this->Latitude, 
    this->Longitude, this->Ve, this->Vn, this->Vu, this->Baseline, this->NSV1, this->NSV2, this->status);
}
