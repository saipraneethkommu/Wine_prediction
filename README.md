# Wine Quality Prediction Model with Spark on AWS

## Overview

This repository contains the code and instructions for building a wine quality prediction model using Apache Spark on the Amazon AWS cloud platform. The model is trained in parallel across multiple EC2 instances to enhance computational efficiency. Docker is employed to streamline model deployment by encapsulating it into a container.

## Table of Contents

- [Description](#description)
- [Setup](#setup)
  - [Cluster Creation](#cluster-creation)
  - [EC2 Instances](#ec2-instances)
  - [S3 Bucket Creation](#s3-bucket-creation)
- [Execution](#execution)
  - [Without Docker](#without-docker-execution)
  - [With Docker](#with-docker-execution)
- [GitHub Repository](#github-repository)
- [Docker Hub](#docker-hub)

## Description

In this individual assignment, the objective is to develop a wine quality prediction machine learning model on the Amazon AWS cloud platform. Utilizing Apache Spark, the model is trained in parallel across EC2 instances to enhance computational efficiency. Spark's MLlib is leveraged for constructing and utilizing the model within the cloud environment. Docker is employed to streamline model deployment by encapsulating it into a container. The entire implementation is carried out in Python on Ubuntu Linux, ensuring compatibility and ease of development.

## Setup

### Cluster Creation

1. Log in to your AWS Account and initiate the lab.
2. Access the AWS Console and search for “EMR".
3. Click on "Create cluster" and assign a name to your cluster.
4. Set the scaling and provisioning parameters: assign 1 instance for core and 5 instances for task.
5. Enable termination protection and choose to manually terminate the cluster under "Cluster termination and node replacement”.
6. Under Security configuration and EC2 key pair, create and assign a key pair.
7. Assign IAM roles: EMR_DefaultRole for Service role and EMR_EC2_DefaultRole for Instance profile.
8. Click on "Create Cluster" to initialize the cluster.
9. Access the cluster details by searching for "EMR", then selecting the created cluster.
10. Connect to the primary node using SSM and note down the Public IP address.

### EC2 Instances

1. Locate the corresponding EC2 instance using the Public IP address.
2. Configure security group inbound rules to allow SSH access from your IP address.
3. Connect to the primary node using SSH.

### S3 Bucket Creation

1. Navigate to the AWS Console and search for "S3".
2. Click on "Create Bucket" and assign a name to your bucket.
3. Upload the required files (working code and datasets) to the created bucket.

## Execution

### Without Docker Execution

1. Navigate to the EC2-connected terminal.
2. Execute the training code: `spark-submit s3://awsbucketwine/Wine_Quality_Training_Spark.py`.
3. Execute the testing code: `spark-submit s3://awsbucketwine/prediction.py`.

### With Docker Execution

1. Set up a Docker repository.
2. Configure the Dockerfile according to your requirements.
3. Make necessary modifications to your code.
4. Check locally and build an image in Docker: `docker build -t train_wine .`.
5. Link the build to the Docker repository: `docker build praneethdocker1/assignment2:praneeth`.
6. Push the image to Docker: `docker push praneethdocker1/assignment2:praneeth`.
7. Execute using Docker: `docker run train_wine`.

## GitHub Repository

Find the code and resources in the [GitHub repository](https://github.com/saipraneethkommu/Wine_prediction).

## Docker Hub

Access the Docker image on [Docker Hub](https://hub.docker.com/repository/docker/praneethdocker1/assignment2/general).
