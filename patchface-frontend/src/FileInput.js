import React from "react";

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
    fetch('localhost:8000/patchface', {
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
        <form onSubmit={this.handleSubmit}>
          <label>
            Upload file:
            <input type="file" ref={this.fileInput} />
          </label>
          <br />
          <button type="submit">Submit</button>
        </form>
        <Image src={`data:image/jpg;base64,${this.state.file}`}/>
      </div>
    );
  }
}