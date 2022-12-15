using Microsoft.MixedReality.Toolkit;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;
using UnityEngine.UI;

public class ImageMsgSubscriber : MonoBehaviour
{

    // Variables required for ROS communication
    [SerializeField]
    string m_TopicName = "/spot/camera/hand_color/compressed";

    public GameObject m_screen;
    public bool is_display_enabled = true;

    // ROS Connector
    private ROSConnection m_Ros;
    private RawImage m_image;
    private Texture2D tex = null;
    private byte[] img_data;
    private bool is_msg_valid = false;

    void Start()
    {
        // Get ROS connection static instance
        CoreServices.SpatialAwarenessSystem.Disable();
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.Subscribe<CompressedImageMsg>(m_TopicName, ImageMsgArrive);
        m_image = m_screen.GetComponent<RawImage>();
        is_msg_valid = false;
        tex = new Texture2D(1, 1);
    }

    private void OnDestroy()
    {
        m_Ros.Disconnect();
    }

    private void ImageMsgArrive(CompressedImageMsg img_msg)
    {
        /*        tex = new Texture2D((int)img_msg.width, (int)img_msg.height, TextureFormat.RGB24, false);*/
        if (!is_msg_valid)
        {
            img_data = img_msg.data;
            is_msg_valid = true;
        }
    }

    void Update()
    {
        if (is_msg_valid && is_display_enabled)
        {
            tex.LoadImage(img_data);
            m_image.texture = tex;
            is_msg_valid = false;
        }
    }
}
