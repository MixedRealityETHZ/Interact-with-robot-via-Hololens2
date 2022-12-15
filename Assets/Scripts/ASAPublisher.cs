using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using System;
using RosMessageTypes.Geometry;

public class ASAPublisher
{
    // Variables required for ROS communication
    [SerializeField]
    string m_TopicName = "/anchor_id";

    // ROS Connector
    public ROSConnection m_Ros;

    public ASAPublisher()
    {
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<StringMsg>(m_TopicName);
    }

/*    public void Publish(List<String> AnchorIDs)*/
    public void Publish(string AnchorIDs)
    {
/*        string[] AnchorIDs_string = AnchorIDs.ToArray();*/
        StringMsg msg = new StringMsg(AnchorIDs);
        m_Ros.Publish(m_TopicName, msg);
    }
}

public class ASASubscriber
{

    // Variables required for ROS communication
    [SerializeField]
    string m_TopicName = "/asa_ros/found_anchor";
    // ROS Connector
    ROSConnection m_Ros;

    // Start is called before the first frame update
    void Start()
    {
        // Get ROS connection static instance
        /*        m_Ros = ROSConnection.GetOrCreateInstance();
                m_Ros.RegisterPublisher<StringMsg>(m_TopicName);
                coroutine = Publisher();
                StartCoroutine(coroutine);*/
    }

    public ASASubscriber()
    {
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterSubscriber(m_TopicName, "PoseStampedMsg");
    }

    /*    public void Publish(List<String> AnchorIDs)*/
    public void Publish(string AnchorIDs)
    {
        /*        string[] AnchorIDs_string = AnchorIDs.ToArray();*/
        StringMsg msg = new StringMsg(AnchorIDs);
        m_Ros.Publish(m_TopicName, msg);
    }

    // Update is called once per frame
    void Update()
    {

    }
}