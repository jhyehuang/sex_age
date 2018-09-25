DROP TABLE deviceid_brand;
DROP TABLE deviceid_package_start_close;
DROP TABLE deviceid_packages;
DROP TABLE deviceid_test;
DROP TABLE deviceid_train;
DROP TABLE package_label;


create table IF NOT EXISTS `deviceid_brand`(
`device_id` varchar(50) not null,
`brand` varchar(50) ,
`type_no` varchar(500) ,
PRIMARY KEY ( `device_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;


create table IF NOT EXISTS `deviceid_package_start_close`(
`id` int(11) NOT NULL AUTO_INCREMENT,
`device_id` varchar(50) not null,
`app_id` varchar(50) ,
`start` varchar(50) ,
`close` varchar(50) ,
`today_hour` varchar(50) ,
`hour_bin` varchar(50) ,
`week` varchar(20) ,
`time_len` int(20) ,
`app_t1` varchar(20) ,
 PRIMARY KEY (`id`)
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE deviceid_package_start_close ADD INDEX app_id_index (`app_id`);
ALTER TABLE deviceid_package_start_close ADD INDEX device_id_index (`device_id`);

create table IF NOT EXISTS `package_label`(
`app_id` varchar(50) not null,
`t1` varchar(50) ,
`t2` varchar(50) ,
PRIMARY KEY ( `app_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE package_label ADD INDEX app_id_index (`app_id`);

create table IF NOT EXISTS `deviceid_packages`(
`device_id` varchar(50) not null,
`add_id_list` text ,
PRIMARY KEY ( `device_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;



create table IF NOT EXISTS `deviceid_test`(
`device_id` varchar(50) not null,
PRIMARY KEY ( `device_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;



create table IF NOT EXISTS `deviceid_train`(
`device_id` varchar(50) not null,
`sex` int(4) not null,
`age` int(4) not null,
PRIMARY KEY ( `device_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;



create table IF NOT EXISTS `proxy_ip_pool`( `protocol` varchar(50) not null, `ip` varchar(50) , `port` varchar(50) , `speed` float(5,3) , `position` varchar(50) , `score` int(20),`failtimes` int(10)  )ENGINE=InnoDB DEFAULT CHARSET=utf8;

