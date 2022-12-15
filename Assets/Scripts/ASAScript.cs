using Microsoft.Azure.SpatialAnchors;
using Microsoft.Azure.SpatialAnchors.Unity;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.Events;
using Microsoft.MixedReality.OpenXR.ARFoundation;
using UnityEngine.UIElements;
using Microsoft.MixedReality.Toolkit.UI;
using RosMessageTypes.Geometry;
using System.Linq;
using Microsoft.MixedReality.Toolkit;
using UnityEngine.SceneManagement;
using static UnityEditor.PlayerSettings;

[RequireComponent(typeof(SpatialAnchorManager))]
public class ASAScript : MonoBehaviour
{
    /// <summary>
    /// Used to distinguish short taps and long taps
    /// </summary>
    private float[] _tappingTimer = { 0, 0 };
    public GameObject attched_gameobject = null;

    public GameObject Arrow = null;

    /// <summary>
    /// Main interface to anything Spatial Anchors related
    /// </summary>
    private SpatialAnchorManager _spatialAnchorManager = null;

    /// <summary>
    /// Used to keep track of all GameObjects that represent a found or created anchor
    /// </summary>
    private List<GameObject> _foundOrCreatedAnchorGameObjects = new List<GameObject>();

    /// <summary>
    /// Used to keep track of all the created Anchor IDs
    /// </summary>
    private List<String> _createdAnchorIDs = new List<String>();

    public GameObject debugBox = null;
    private TextMesh textBox = null;

    private ASAPublisher anchor_publisher = null;
    [SerializeField]
    public float wating_sec = 3.0F;

    public float _timer = -1f;
    private Vector3 target_pos;

    // <Start>
    // Start is called before the first frame update
    void Start()
    {
        CoreServices.SpatialAwarenessSystem.Disable();
        textBox = debugBox.GetComponent<TextMesh>();
        textBox.text = "Move the sphere to \nand the robot will \ngo there";
        _spatialAnchorManager = GetComponent<SpatialAnchorManager>();
        _spatialAnchorManager.LogDebug += (sender, args) => Debug.Log($"ASA - Debug: {args.Message}");
        _spatialAnchorManager.Error += (sender, args) => Debug.LogError($"ASA - Error: {args.ErrorMessage}");
        _spatialAnchorManager.AnchorLocated += SpatialAnchorManager_AnchorLocated;
        anchor_publisher = new ASAPublisher();
    }
    // </Start>
    private void OnDestroy()
    {
        anchor_publisher.m_Ros.Disconnect();
    }

    public void ChangeScene(int i)
    {
        if (i == 0)
        {
        }
        else if(i == 1)
        {
            SceneManager.LoadScene(sceneName: "HandGestureSpot");
        }
        else if(i == 2)
        {
            SceneManager.LoadScene(sceneName: "RotationCube");
        }
    }


    private void OnDestroy()
    {
/*        DeleteAnchor(_createdAnchorIDs);*/
    }

    // <Update>
    // Update is called once per frame
    void Update()
    {
        //Check for any air taps from either hand
        for (int i = 0; i < 2; i++)
        {
            InputDevice device = InputDevices.GetDeviceAtXRNode((i == 0) ? XRNode.RightHand : XRNode.LeftHand);
            if (device.TryGetFeatureValue(CommonUsages.primaryButton, out bool isTapping))
            {
                if (!isTapping)
                {
                    //Stopped Tapping or wasn't tapping
                    if (0f < _tappingTimer[i] && _tappingTimer[i] < 1f && !attched_gameobject.GetComponent<AnchorManipulator>().is_on_manipulation)
                    {
                        //User has been tapping for less than 1 sec. Get hand position and call ShortTap
                        _timer = -1f;
                    }
                    _tappingTimer[i] = 0;
                }
                else
                {
                    _tappingTimer[i] += Time.deltaTime;
                    if (_tappingTimer[i] >= 2f && !attched_gameobject.GetComponent<AnchorManipulator>().is_on_manipulation)
                    {
                        //User has been air tapping for at least 2sec. Get hand position and call LongTap
                        if (device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 handPosition))
                        {
                            LongTap();
                        }
                        _tappingTimer[i] = -float.MaxValue; // reset the timer, to avoid retriggering if user is still holding tap
                    }
                }
            }
        }
        if (_timer > wating_sec)
        {
            _timer = -1f;
            NewAnchorPos(attched_gameobject.transform.position);
        }
        else if(_timer < 0f) { }
        else
        {
            _timer += Time.deltaTime;
            textBox.text = "Placing anchor in " + Math.Ceiling(wating_sec - _timer).ToString() + " seconds";
        }
    }
    // </Update>

    // <LongTap>
    /// <summary>
    /// Called when a user is air tapping for a long time (>=2 sec)
    /// </summary>
    private async void LongTap()
    {
        if (_spatialAnchorManager.IsSessionStarted)
        {
            // Stop Session and remove all GameObjects. This does not delete the Anchors in the cloud
            _spatialAnchorManager.DestroySession();
            RemoveAllAnchorGameObjects();
            Arrow.SetActive(false);
            textBox.text = ("ASA - Stopped Session and removed all Anchor Objects");
        }
        else
        {
            //Start session and search for all Anchors previously created
            await _spatialAnchorManager.StartSessionAsync();
            LocateAnchor();
        }
    }
    // </LongTap>

    // <RemoveAllAnchorGameObjects>
    /// <summary>
    /// Destroys all Anchor GameObjects
    /// </summary>
    private void RemoveAllAnchorGameObjects()
    {
        foreach (var anchorGameObject in _foundOrCreatedAnchorGameObjects)
        {
            Destroy(anchorGameObject);
        }
        _foundOrCreatedAnchorGameObjects.Clear();
    }


    // <NewAnchorPos>
    /// <summary>
    /// Called when a user is air tapping for a short time 
    /// </summary>
    /// <param name="handPosition">Location where tap was registered</param>
    private async void NewAnchorPos(Vector3 handPosition)
    {
        await _spatialAnchorManager.StartSessionAsync();
        //No Anchor Nearby, start session and create an anchor
        //Delete old Anchor
/*        foreach (var anchor in _foundOrCreatedAnchorGameObjects)
        {
            DeleteAnchor(anchor);
        }
        _createdAnchorIDs.Clear();*/
        await CreateAnchor(handPosition);
    }
    // </NewAnchorPos>

    // <CreateAnchor>
    /// <summary>
    /// Creates an Azure Spatial Anchor at the given position rotated towards the user
    /// </summary>
    /// <param name="position">Position where Azure Spatial Anchor will be created</param>
    /// <returns>Async Task</returns>
    private async Task CreateAnchor(Vector3 position)
    {
        textBox.text = "Creating the anchor...";
        //Create Anchor GameObject. We will use ASA to save the position and the rotation of this GameObject.
        if (!InputDevices.GetDeviceAtXRNode(XRNode.Head).TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 headPosition))
        {
            headPosition = Vector3.zero;
        }

        Quaternion orientationTowardsHead = Quaternion.LookRotation(position - headPosition, Vector3.up);

        /*        GameObject anchorGameObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);*/
        GameObject anchorGameObject = new GameObject();
        /*        anchorGameObject.AddComponent<MeshRenderer>().material.shader = Shader.Find("Legacy Shaders/Diffuse");*/
        anchorGameObject.transform.position = position;
        anchorGameObject.transform.rotation = Quaternion.Euler(0, 90, 0) * orientationTowardsHead;

        //Add and configure ASA components
        CloudNativeAnchor cloudNativeAnchor = anchorGameObject.AddComponent<CloudNativeAnchor>();
        await cloudNativeAnchor.NativeToCloud();
        CloudSpatialAnchor cloudSpatialAnchor = cloudNativeAnchor.CloudAnchor;
        cloudSpatialAnchor.Expiration = DateTimeOffset.Now.AddDays(1);

        //Collect Environment Data
        while (!_spatialAnchorManager.IsReadyForCreate)
        {
            float createProgress = _spatialAnchorManager.SessionStatus.RecommendedForCreateProgress;
            textBox.text = ($"ASA - Move your device to capture more environment data: {createProgress:0%}");
        }

        try
        {
            // Now that the cloud spatial anchor has been prepared, we can try the actual save here.
            await _spatialAnchorManager.CreateAnchorAsync(cloudSpatialAnchor);

            bool saveSucceeded = cloudSpatialAnchor != null;
            if (!saveSucceeded)
            {
                Debug.LogError("ASA - Failed to save, but no exception was thrown.");
                return;
            }

            _foundOrCreatedAnchorGameObjects.Add(anchorGameObject);
            _createdAnchorIDs.Add(cloudSpatialAnchor.Identifier);
            anchor_publisher.Publish(cloudSpatialAnchor.Identifier);
            Arrow.SetActive(true);
            Arrow.transform.position = position;
            Arrow.transform.rotation = orientationTowardsHead;
            textBox.text = "Anchor created!";
            /*            anchorGameObject.GetComponent<MeshRenderer>().material.color = Color.green;*/
        }
        catch (Exception exception)
        {
            textBox.text = ("ASA - Failed to save anchor: " + exception.ToString());
            Debug.LogException(exception);
        }
    }
    // </CreateAnchor>

    // <LocateAnchor>
    /// <summary>
    /// Looking for anchors with ID in _createdAnchorIDs
    /// </summary>
    private void LocateAnchor()
    {
        if (_createdAnchorIDs.Count > 0)
        {
            //Create watcher to look for all stored anchor IDs
            textBox.text = ($"ASA - Creating watcher to look for {_createdAnchorIDs.Count} spatial anchors");
            AnchorLocateCriteria anchorLocateCriteria = new AnchorLocateCriteria();
            anchorLocateCriteria.Identifiers = _createdAnchorIDs.ToArray();
            _spatialAnchorManager.Session.CreateWatcher(anchorLocateCriteria);
            textBox.text = ($"ASA - Watcher created!");
        }
    }
    // </LocateAnchor>

    // <SpatialAnchorManagerAnchorLocated>
    /// <summary>
    /// Callback when an anchor is located
    /// </summary>
    /// <param name="sender">Callback sender</param>
    /// <param name="args">Callback AnchorLocatedEventArgs</param>
    private void SpatialAnchorManager_AnchorLocated(object sender, AnchorLocatedEventArgs args)
    {
/*        textBox.text = ($"ASA - Anchor recognized as a possible anchor {args.Identifier} {args.Status}");*/

        if (args.Status == LocateAnchorStatus.Located)
        {
            //Creating and adjusting GameObjects have to run on the main thread. We are using the UnityDispatcher to make sure this happens.
            UnityDispatcher.InvokeOnAppThread(() =>
            {
                // Read out Cloud Anchor values
                CloudSpatialAnchor cloudSpatialAnchor = args.Anchor;

                //Create GameObject
                GameObject anchorGameObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                anchorGameObject.transform.localScale = Vector3.one * 0.05f;
                anchorGameObject.GetComponent<MeshRenderer>().material.shader = Shader.Find("Legacy Shaders/Diffuse");
                anchorGameObject.GetComponent<MeshRenderer>().material.color = Color.green;

                // Link to Cloud Anchor
                anchorGameObject.AddComponent<CloudNativeAnchor>().CloudToNative(cloudSpatialAnchor);
                _foundOrCreatedAnchorGameObjects.Add(anchorGameObject);
                if (args.Identifier == _createdAnchorIDs.Last())
                {
                    textBox.text = "Last added anchor position\n" + anchorGameObject.transform.position.ToString() + "\n" + 
                    anchorGameObject.transform.rotation.ToString();
                }
            });
        }
    }
    // </SpatialAnchorManagerAnchorLocated>

    // <DeleteAnchor>
    /// <summary>
    /// Deleting Cloud Anchor attached to the given GameObject and deleting the GameObject
    /// </summary>
    /// <param name="anchorGameObject">Anchor GameObject that is to be deleted</param>
    private async void DeleteAnchor(GameObject anchorGameObject)
    {
        CloudNativeAnchor cloudNativeAnchor = anchorGameObject.GetComponent<CloudNativeAnchor>();
        CloudSpatialAnchor cloudSpatialAnchor = cloudNativeAnchor.CloudAnchor;
/*
        textBox.text = ($"ASA - Deleting cloud anchor: {cloudSpatialAnchor.Identifier}");*/

        //Request Deletion of Cloud Anchor
        await _spatialAnchorManager.DeleteAnchorAsync(cloudSpatialAnchor);

        //Remove local references
        _createdAnchorIDs.Remove(cloudSpatialAnchor.Identifier);
        _foundOrCreatedAnchorGameObjects.Remove(anchorGameObject);
        Destroy(anchorGameObject);
    }

}