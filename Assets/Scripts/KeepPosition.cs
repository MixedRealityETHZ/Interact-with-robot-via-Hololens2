using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KeepPosition : MonoBehaviour
{

    public Vector3 pos = new Vector3(0,0,1);
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        gameObject.transform.position = Camera.main.transform.position
            + pos.x * Camera.main.transform.right
            + pos.y * Camera.main.transform.up
            + pos.z * Camera.main.transform.forward;
        gameObject.transform.rotation = Camera.main.transform.rotation;
    }
}
