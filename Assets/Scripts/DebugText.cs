using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DebugText : MonoBehaviour
{
    [SerializeField]
    TextMesh textMesh;
    // Start is called before the first frame update
    void Start()
    {
        textMesh = gameObject.GetComponent<TextMesh>();
/*        textMesh.text = "Debug Start";*/
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void ChangeText(string str)
    {
        textMesh.text = str;
    }
    
    public void AppendText(string str)
    {
        textMesh.text += "\n" + str;
    }
    public void ClearText(string str)
    {
        textMesh.text = "";
    }

}
