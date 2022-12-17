using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class FollowCamera : MonoBehaviour
{

    public Vector3 position = new Vector3(0, 0, 1);
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        gameObject.transform.position = position + Camera.main.transform.position;
        gameObject.transform.rotation = Camera.main.transform.rotation;
    }
}
