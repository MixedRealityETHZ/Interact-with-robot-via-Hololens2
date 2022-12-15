using UnityEngine;
using RosMessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector;
using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System;
using UnityEngine.Events;
using Microsoft.MixedReality.Toolkit;
using UnityEngine.SceneManagement;

public class CubeManipulator : MonoBehaviour
{

    public GameObject Cube;
    public ImageMsgSubscriber img_sub;
    public Vector3 m_cube_init_trans = new Vector3(-0.1F, 0, 0.5F);
    private Quaternion m_cube_init_rotate = new Quaternion(0, 0, 0, 1.0F);
    public float m_recover_rate = 1.0F;
    public bool is_fixed_cube2world = true;
    public float volocity = 0.2f;

    // Variables required for ROS communication
    [SerializeField]
    string m_TopicName = "/spot/cmd_vel";

    // ROS Connector
    ROSConnection m_Ros;

    private bool is_on_manipulation = false;
    private float rate = 0.125F;
    private Vector3 pos_maniplate_start;
    private Quaternion rot_maniplate_start;
    private IEnumerator coroutine;

    void Start()
    {
        // Get ROS connection static instance
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<TwistMsg>(m_TopicName);
        coroutine = Publisher();
        StartCoroutine(coroutine);
    }
    public void FixCube()
    {
        is_fixed_cube2world = !is_fixed_cube2world;
    }

    private void OnDestroy()
    {
        m_Ros.Disconnect();
    }

    public void ChangeScene(int i)
    {
        if (i == 0)
        {
            SceneManager.LoadScene(sceneName: "AzureSpatialSpot");
        }
        else if (i == 1)
        {
            SceneManager.LoadScene(sceneName: "HandGestureSpot");
        }
        else if (i == 2)
        {
        }
    }

    public void SetRate(SliderEventData eventData)
    {
        volocity = eventData.NewValue;
    }
    public void Publish(TwistMsg twist)
    {
        m_Ros.Publish(m_TopicName, twist);
    }
    private IEnumerator Publisher()
    {
        while (true)
        {
            if (is_on_manipulation)
            {
                var delta_trans = Cube.transform.localPosition - pos_maniplate_start;
                // set dist as velocity
                delta_trans.z = 0;
                var dist = delta_trans.sqrMagnitude;
                if (dist < 0.0009f)
                {
                    delta_trans.x = delta_trans.y = 0;
                }
                else if(dist > 0.01f)
                {
                    delta_trans.x = delta_trans.normalized.x * 0.1f;
                    delta_trans.y = delta_trans.normalized.y * 0.1f;
                }
                delta_trans.x = Mathf.Round(delta_trans.x / 0.02f) * 0.02f;
                delta_trans.y = Mathf.Round(delta_trans.y / 0.02f) * 0.02f;
                var delta_rot = Quaternion.ToEulerAngles(Quaternion.Inverse(rot_maniplate_start) * Cube.transform.localRotation);
                if(delta_rot.y > -0.5f && delta_rot.y < 0.5f)
                {
                    delta_rot.y = 0;
                }
                else if (delta_rot.y >= 0.5f && delta_rot.y < 1.0f)
                {
                    delta_rot.y -= 0.5f;
                }
                else if (delta_rot.y >= -1.0f && delta_rot.y <= -0.5f)
                {
                    delta_rot.y += 0.5f;
                }
                delta_rot.y = Mathf.Round(delta_rot.y / 0.05f) * 0.05f;
                delta_rot.y = Mathf.Clamp(delta_rot.y, -0.7f, 0.7f);
                var twist = new TwistMsg();
                twist.angular = new Vector3Msg(0, 0, -delta_rot.y);
                twist.linear = new Vector3Msg(delta_trans.y / 0.1f * volocity, -delta_trans.x / 0.1f * volocity, 0);
                Publish(twist);
            }
            yield return new WaitForSeconds(rate);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (!is_on_manipulation)
        {
            if (is_fixed_cube2world)
            {
                var translate = m_cube_init_trans - Cube.transform.localPosition;
                if (translate.sqrMagnitude < 1e-8F)
                {
                    Cube.transform.localPosition = m_cube_init_trans;
                }
                else
                {
                    Cube.transform.localPosition += translate * m_recover_rate;
                }
                Cube.transform.localRotation = Quaternion.RotateTowards(Cube.transform.localRotation, m_cube_init_rotate, m_recover_rate * 80);
            }
            else
            {
                var translate = m_cube_init_trans - Cube.transform.position;
                if (translate.sqrMagnitude < 1e-8F)
                {
                    Cube.transform.position = m_cube_init_trans;
                }
                else
                {
                    Cube.transform.position += translate * m_recover_rate;
                }
                Cube.transform.rotation = Quaternion.RotateTowards(Cube.transform.rotation, m_cube_init_rotate, m_recover_rate * 80);
            }
        }
    }

    public void OnCubeMove(ManipulationEventData eventData)
    {
        if (eventData.Pointer != null)
        {
            pos_maniplate_start = Cube.transform.localPosition;
            rot_maniplate_start = Cube.transform.localRotation;
            is_on_manipulation = true;
        }
    }

    public void OnCubeStop(ManipulationEventData eventData)
    {
        is_on_manipulation = false;
    }
}
