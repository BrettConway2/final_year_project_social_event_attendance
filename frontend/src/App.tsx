import { useState } from 'react'
import './App.css'

import Results from './Results';
import { Container, Overlay, Spinner, type Person } from './styles';
import Upload from './Upload';





function App() {
  const [people, setPeople] = useState<Person[]>([]);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);



  return (
    <Container>
      {loading ? (
        <Overlay>
          <Spinner />
        </Overlay>
        
      ) : (
        ready ? (<Results people={people} loading={loading} setReady={setReady} />) : (<Upload setReady={setReady} setLoading={setLoading} setPeople={setPeople}/>)    
      )}
    </Container>
  );
}

export default App;
