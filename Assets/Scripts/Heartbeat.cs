using RosMessageTypes.Std;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using System;

public class Heartbeat : MonoBehaviour
{
    string m_TopicName = "/alive";
    ROSConnection m_Ros;
    private IEnumerator coroutine;
    private float rate = 1.0F;

    // Start is called before the first frame update
    void Start()
    {
        // Get ROS connection static instance
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<StringMsg>(m_TopicName);
        coroutine = Publisher();
        StartCoroutine(coroutine);
    }

    private IEnumerator Publisher()
    {
        while (true)
        {
            StringMsg msg = new StringMsg("Alive! " + DateTime.Now.ToString("h:mm:ss tt"));
            m_Ros.Publish(m_TopicName, msg);
            yield return new WaitForSeconds(rate);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
