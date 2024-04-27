FROM centos:7

RUN yum -y update && yum -y install python3 python3-pip java-1.8.0-openjdk wget unzip
RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install awscli
RUN pip3 install numpy pandas

WORKDIR /opt

RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" \
  && tar -xf apache-spark.tgz \
  && rm apache-spark.tgz \
  && ln -s spark-3.5.0-bin-hadoop3 spark

RUN yum -y install fuse-overlayfs

ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.8.0/aws-java-sdk-1.8.0.jar -P /spark/jars/
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.0/hadoop-aws-3.0.0.jar -P /spark/jars/

ADD train_wine.py .
ADD TrainingDataset.csv .
ADD ValidationDataset.csv .
ENTRYPOINT ["/opt/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:2.7.3,com.amazonaws:aws-java-sdk:1.7.4", "train_wine.py"]
