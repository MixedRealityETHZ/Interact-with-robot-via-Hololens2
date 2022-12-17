using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEngine.Serialization;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit;
using System.Reflection;
using System;
using TMPro;
using System.IO;
using Unity.Barracuda;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using System.Threading;
using UnityEngine.SceneManagement;

enum HandGesture
{
    TurnLeft,
    TurnRight,
    Forward,
    Stop,
    Nothing
}

public class HandJointExport : MonoBehaviour
{

    string OverallLogging = "";
    public Camera camera;
    public TMP_Text LoggingPanel;
    public TMP_Text WarningPanel;
    int tensoridx = 0;
    string action_text = "";

    Tensor inputTensor = new Tensor(1, 78*5);

    [SerializeField]

    private NNModel kerasModel;


    private Vector3 cameraPosition;
    private Vector3 cameraEulerAngles;
    private Quaternion cameraRotation;

    private Vector3 cameraRealtimePosition;
    private Vector3 cameraRealtimeEulerAngles;
    private Quaternion cameraRealtimeRotation;

    private Vector3 RelativePosition;
    private Vector3 RelativeEulerAngles;
    private Quaternion RelativeRotation;
    private float countdown = -10000;

    private Model runtimeModel;

    private IWorker worker;

    private string outputLayerName;

    [SerializeField]
    string m_TopicName = "/spot/cmd_vel";

    // ROS Connector
    ROSConnection m_Ros;
    // 
    private HandGesture hand_status = HandGesture.Nothing;
    private int timer = 8;
    private float rate = 0.125F;
    private IEnumerator coroutine;

    public void ChangeScene(int i)
    {
        if (i == 0)
        {
            SceneManager.LoadScene(sceneName: "AzureSpatialSpot");
        }
        else if (i == 1)
        {
        }
        else if (i == 2)
        {
            SceneManager.LoadScene(sceneName: "RotationCube");
        }
    }
    public void Publish(TwistMsg twist)
    {
        m_Ros.Publish(m_TopicName, twist);
    }

    private void OnDestroy()
    {
        m_Ros.Disconnect();
    }

    private IEnumerator Publisher()
    {
        while (true)
        {
            var twist = new TwistMsg();
            twist.angular = new Vector3Msg(0, 0, 0);
            switch (hand_status)
            {
                case HandGesture.TurnLeft:
                    twist.linear = new Vector3Msg(0, -0.4, 0);
                    Publish(twist);
                    break;
                case HandGesture.TurnRight:
                    twist.linear = new Vector3Msg(0, 0.4, 0);
                    Publish(twist);
                    break;
                case HandGesture.Forward:
                    twist.linear = new Vector3Msg(0.4, 0, 0);
                    Publish(twist);
                    break;
                case HandGesture.Stop:
                    twist.linear = new Vector3Msg(-0.4, 0, 0);
                    Publish(twist);
                    break;
                default:
                    break;
            }
            yield return new WaitForSeconds(rate);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        cameraPosition = camera.transform.position;
        cameraRotation = camera.transform.rotation;

        cameraEulerAngles = camera.transform.eulerAngles;
        runtimeModel = ModelLoader.Load(kerasModel);

        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);
            
        outputLayerName = runtimeModel.outputs[runtimeModel.outputs.Count - 1];


        // Get ROS connection static instance
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterPublisher<TwistMsg>(m_TopicName);
        coroutine = Publisher();
        StartCoroutine(coroutine);
        hand_status = HandGesture.Nothing;
        timer = 0;
        countdown = 0;
    }

    public float Sigmoid(float value) {
        float k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    public int GetMaxArrayElement(float[] array)
    {
        int index = -1;
        float max = int.MinValue;
        for (int i = 0; i < array.Length; ++i)
        {
            if (array[i] > max)
            {
                max = array[i];
                index = i;
            }
        }
        
        return index;
    }

    private void FixedUpdate()
    {
        cameraRealtimePosition = camera.transform.position;
        cameraRealtimeRotation = camera.transform.rotation;
        cameraRealtimeEulerAngles = camera.transform.eulerAngles;
        RelativeEulerAngles = cameraRealtimeEulerAngles - cameraEulerAngles;
        RelativePosition = cameraRealtimePosition - cameraPosition;
        RelativeRotation = new Quaternion( cameraRealtimeRotation.x-cameraRotation.x, cameraRealtimeRotation.y - cameraRotation.y, cameraRealtimeRotation.z - cameraRotation.z, cameraRealtimeRotation.w - cameraRotation.w);

        if(countdown < 5)
        {
            countdown += Time.deltaTime;
            WarningPanel.text = "Hand Gesture Recognization will start in " + Math.Ceiling(5 - countdown).ToString() + " seconds";
        }
        else
        {
            if (IsInvalidTracking())
            {
                WarningPanel.text = "hand found";
                RefreshTrackedObject();
            }
            else
            {
                tensoridx = 0;
                WarningPanel.text = "hand not found";
            }
        }
    }

    private GameObject trackingTarget;
    // Stores currently attached hand if valid (only possible values Left, Right, or None)
    protected Handedness currentTrackedHandedness = Handedness.None;
    [SerializeField]
    [Tooltip("If tracking hands or motion controllers, determines which hand(s) are valid attachments")]
    [FormerlySerializedAs("trackedHandness")]
    private Handedness trackedHandedness = Handedness.Left;

    private bool IsInvalidTracking()
    {
        if (trackingTarget == null || trackingTarget.transform.parent == null)
        {
            return true;
        }
        // If we were tracking a particular hand, check that our transform is still valid
        // The HandJointService does not destroy its own hand joint tracked GameObjects even when a hand is no longer tracked
        // Those HandJointService's GameObjects though are the parents of our tracked transform and thus will not be null/destroyed
        if (!currentTrackedHandedness.IsNone())
        {
            bool trackingLeft = HandJointService.IsHandTracked(Handedness.Left);
            bool trackingRight = HandJointService.IsHandTracked(Handedness.Right);

            return (currentTrackedHandedness.IsLeft() && !trackingLeft) ||
                   (currentTrackedHandedness.IsRight() && !trackingRight);
        }

        return false;
    }
    //[SerializeField]
    //[Tooltip("When TrackedTargetType is set to hands, use this specific joint to calculate position and orientation")]
    //private TrackedHandJoint trackedHandJoint = TrackedHandJoint.Palm;
    /// <summary>
    /// When TrackedTargetType is set to hands, use this specific joint to calculate position and orientation
    /// </summary>
    //public TrackedHandJoint TrackedHandJoint
    //{
    //    get => trackedHandJoint;
    //    set
    //    {
    //        if (trackedHandJoint != value)
    //        {
    //            trackedHandJoint = value;
    //            RefreshTrackedObject();
    //        }
    //    }
    //}
    /// <summary>
    /// If tracking hands or motion controllers, determines which hand(s) are valid attachments.
    /// </summary>
    /// <remarks>
    /// Only None, Left, Right, and Both are valid values
    /// </remarks>
    public Handedness TrackedHandedness
    {
        get => trackedHandedness;
        set
        {
            if (trackedHandedness != value)
            {
                trackedHandedness = value;
                RefreshTrackedObject();
            }
        }
    }


    /// <summary>
    /// Currently tracked hand or motion controller if applicable
    /// </summary>
    /// <remarks>
    /// Only possible values Left, Right, or None
    /// </remarks>
    public Handedness CurrentTrackedHandedness => currentTrackedHandedness;

    // Stores controller side to favor if TrackedHandedness is set to both
    protected Handedness preferredTrackedHandedness = Handedness.Left;
    /// <summary>
    /// Controller side to favor and pick first if TrackedHandedness is set to both
    /// </summary>
    /// <remarks>
    /// Only possible values, Left or Right
    /// </remarks>
    public Handedness PreferredTrackedHandedness
    {
        get => preferredTrackedHandedness;
        set
        {
            if ((value.IsLeft() || value.IsRight())
                && preferredTrackedHandedness != value)
            {
                preferredTrackedHandedness = value;
            }
        }
    }

    private IMixedRealityHandJointService HandJointService
    => handJointService ?? (handJointService = CoreServices.GetInputSystemDataProvider<IMixedRealityHandJointService>());

    private IMixedRealityHandJointService handJointService = null;

    public void RefreshTrackedObject()
    {
        DetachFromCurrentTrackedObject();
        AttachToNewTrackedObject();
    }
    protected virtual void DetachFromCurrentTrackedObject()
    {
        if (trackingTarget != null)
        {
            trackingTarget.transform.parent = null;
        }
    }
    protected virtual void AttachToNewTrackedObject()
    {
        currentTrackedHandedness = Handedness.None;
        Transform target = null;
        if (HandJointService != null)
        {
            currentTrackedHandedness = TrackedHandedness;
            if (currentTrackedHandedness == Handedness.Both)
            {
                if (HandJointService.IsHandTracked(PreferredTrackedHandedness))
                {
                    currentTrackedHandedness = PreferredTrackedHandedness;
                }
                else if (HandJointService.IsHandTracked(PreferredTrackedHandedness.GetOppositeHandedness()))
                {
                    currentTrackedHandedness = PreferredTrackedHandedness.GetOppositeHandedness();
                }
                else
                {
                    currentTrackedHandedness = Handedness.None;
                }
            }
            string log = "";
            int joint_idx = 0;
            // TODO: !!!! add bounding value
            float xmin = -0.4f;
            float xmax = 0.4f;
            float ymin = -0.40f;
            float ymax = 0.15f;
            float zmin = -0.0f;
            float zmax = 0.7f;
            bool out_of_box = false;
            foreach (TrackedHandJoint trackedHandJoint in Enum.GetValues(typeof(TrackedHandJoint)))
            {
                joint_idx += 1;
                if (trackedHandJoint != 0)
                {
                    target = HandJointService.RequestJointTransform(trackedHandJoint, currentTrackedHandedness);
                    if (target) { 
                        target.parent = camera.transform;
                        //Debug.Log("posBEFORE" + target.localPosition.ToString());
                        //Debug.Log("RelativeEulerAngles" + RelativeEulerAngles.ToString());
                        //target.transform.Rotate(-RelativeEulerAngles, Space.Self);
                        //target.transform.position -= RelativePosition;
                        ////target.transform.rotation = new Quaternion(target.transform.rotation.x - RelativeRotation.x, target.transform.rotation.y - RelativeRotation.y, target.transform.rotation.z - RelativeRotation.z, target.transform.rotation.w - RelativeRotation.w);
                        //target.transform.Rotate(RelativeEulerAngles, Space.Self);
                        log += target.transform.position.ToString();

                        Vector3 position = target.localPosition;
                        //Debug.Log("pos" + target.localPosition.ToString());
                        // Debug.Log(tensoridx);
                        try
                        {
                            inputTensor[tensoridx] = position[0];
                            tensoridx += 1;
                            inputTensor[tensoridx] = position[1];
                            tensoridx += 1;
                            inputTensor[tensoridx] = position[2];
                            tensoridx += 1;
                            if (joint_idx == 10){
                                if (!( (xmin < position[0]) && (position[0]<xmax) && (ymin < position[1]) && (position[1] < ymax) && (zmin < position[2]) && (position[2] < zmax)   )) {
                                    hand_status = HandGesture.Nothing;
                                    tensoridx = 0;
                                    out_of_box = true;
                                }
                            }
                        }
                        catch (Exception e){
                            //pass
                        }
                    }    
                }
            }
            
            if (tensoridx==78*5) {
                worker.Execute(inputTensor);
                Tensor outputTensor = worker.PeekOutput(outputLayerName);
                //Debug.Log(outputTensor);
                // float[] outputlist = new float[] { Sigmoid(outputTensor[0]), Sigmoid(outputTensor[1]), Sigmoid(outputTensor[2]), Sigmoid(outputTensor[3]) };
                float[] outputlist = new float[] {outputTensor[0], outputTensor[1], outputTensor[2], outputTensor[3] };
                int argmax;
                argmax = GetMaxArrayElement(outputlist);
                if (argmax == 0)
                {
                    hand_status = HandGesture.TurnRight;
                    action_text = "TurnRight";
                }
                else if (argmax == 1)
                {
                    hand_status = HandGesture.TurnLeft;
                    action_text = "TurnLeft";
                }
                else if (argmax == 2)
                {
                    hand_status = HandGesture.Forward;
                    action_text = "WalkForward";
                }
                else if (argmax == 3)
                {
                    hand_status = HandGesture.Stop;
                    action_text = "Stop";
                }

                tensoridx = 0;
                //Debug.Log(log);
                LoggingPanel.text = "Prediction:" + action_text + " " + string.Join(",", outputlist) + "\r\n" + log;
                OverallLogging += log;
                OverallLogging += "\r\n";
            }
            if (log.Length > 0 && out_of_box == false)
            {
                WarningPanel.text = "Collecting data: " + tensoridx + "/390";
            } else
            {
                if (out_of_box)
                {
                    WarningPanel.text = "Hand out of bound!";
                }
                else {
                    WarningPanel.text = "Hand not found!";
                }
                LoggingPanel.text = "Prediction: None";
                tensoridx = 0;
                hand_status = HandGesture.Nothing;
                
            }
            // if (log.Length > 0)
            // {
                
            // }
        }
    }
}
