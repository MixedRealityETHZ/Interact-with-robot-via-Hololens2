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
using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;

public class AnchorManipulator : MonoBehaviour
{
    public bool is_on_manipulation = false;
    public GameObject ASAController = null;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnAnchorMove(ManipulationEventData eventData)
    {
        if (eventData.Pointer != null)
        {
            is_on_manipulation = true;
            ASAController.GetComponent<ASAScript>()._timer = -1f;
        }
    }

    public void OnAnchorStop(ManipulationEventData eventData)
    {
        is_on_manipulation = false;
        ASAController.GetComponent<ASAScript>()._timer = 0f;
    }
}
