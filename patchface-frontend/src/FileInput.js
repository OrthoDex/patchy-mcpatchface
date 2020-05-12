import React from "react";
import './FileInput.css'
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Image from 'react-bootstrap/Image';

export class FileInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.fileInput = React.createRef();
    this.state = {
      file: []
    }
  }
  handleSubmit(event) {
    event.preventDefault();
    fetch('http://localhost:8000/patchface', {
      method: 'POST',
      fileInput: this.fileInput.current.files[0]
    }).then(resp => {
      if (resp.ok) {
        this.setState({
          file: resp.blob
        })
        alert('done')
      } else {
        alert('error')
      }
    }).catch(e => {
      alert(e)
    });
  }

  render() {
    return (
      <div id='container' >
        <Form onSubmit={this.handleSubmit}>
          <Form.Label>
            Upload file:
            <Form.Control type="file" ref={this.fileInput} />
          </Form.Label>
          <br />
          <Button type="submit">Submit</Button>
        </Form>
        {this.state.file.length !== 0 ? <Image alt='image file' src={`data:image/jpg;base64,${this.state.file}`}/> : <p></p>}
      </div>
    );
  }
}