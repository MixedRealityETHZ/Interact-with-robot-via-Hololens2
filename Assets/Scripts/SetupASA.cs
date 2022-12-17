using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetupASA : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        GameObject.Find("Cursor").SetActive(true);
        GameObject.Find("textBox").GetComponent<MeshRenderer>().enabled = true;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
