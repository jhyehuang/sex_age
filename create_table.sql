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
`start_day` varchar(50) ,
`close_day` varchar(50) ,
`start_day_cnt` int(20) ,
`close_day_cnt` int(20) ,
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

