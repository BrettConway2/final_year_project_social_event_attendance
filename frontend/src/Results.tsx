import './App.css'

import { AppearanceFigure, AppearanceScroll, CloseButton, Container, FaceImage, FetchButton, ImageContainer, Modal, ModalImage, NavButton, PersonDetails, ScrollImage, ScrollItems, Spinner } from './styles';
import { useEffect, useState } from 'react';
import { ResultsPhotoScroller } from './ResultsPhotoScroller';

type Wardrobe = { hair: string;
                  skin: string;  
                  shirt_dress1: string;
                  shirt_dress2: string;
                  shirt_dress3: string;
                  shirt_dress4: string;
                  shirt_dress5: string;
                  pants1: string;
                  pants2: string;
                  pants3: string;
                  pants4: string;
                  pants5: string;
                  skirt1: string;
                  skirt2: string;
                  skirt3: string;
                  skirt4: string;
                  skirt5: string;
                };
type Appearance = { photo_num: number; image: string };
type Person = { description: Wardrobe; face: string; appearances: Appearance[] };

type ResultsProps = {
  people: Person[];
  loading: boolean;
  setReady: React.Dispatch<React.SetStateAction<boolean>>;
};


type NameProps = {
  initialName: string;
};

const ColorBox = ({ color }: { color: string }) => (
  <div
    style={{
      width: '30px',
      height: '30px',
      backgroundColor: color,
      border: '1px solid #ccc',
      borderRadius: '4px',
    }}
  />
);


const WardrobeDisplay = ({ wardrobe }: { wardrobe: Wardrobe }) => {

  const grouped: Record<string, string[]> = {};

  for (const [key, color] of Object.entries(wardrobe)) {
    const base = key.replace(/[0-9]+$/, '');
    if (!grouped[base]) grouped[base] = [];
    grouped[base].push(color);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {Object.entries(grouped).map(([base, colors]) => (
        <div key={base} style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
          <span style={{ width: '120px', textTransform: 'capitalize' }}>
            {base.replace('_', '/')}:
          </span>
          {colors.map((color, idx) => (
            <ColorBox key={idx} color={color} />
          ))}
        </div>
      ))}
    </div>
  );
};


function EditableName({initialName}: NameProps) {
  const [name, setName] = useState(initialName);

  return (
    <div style={{display: 'flex', flexDirection: 'row', justifyContent: 'left', alignContent: 'center', alignItems: 'center', gap: '10px'}}>
      <div style={{fontWeight: 'bold', display: 'flex', alignItems: 'center'}}>Name:</div>
      <input
        id="name"
        type="text"
        value={name}
        onChange={(e) => setName(e.target.value)}
        style={{
          padding: '8px',
          fontSize: '16px',
          borderRadius: '4px',
          border: '1px solid #ccc',
        }}
      />
    </div>
  );
}




function Results({people, loading, setReady}: ResultsProps) {
  const [images, setImages] = useState<string[]>([]);
  const[imagesLoading, setImagesLoading] = useState<boolean>(true)


  const [modalOpen, setModalOpen] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);


  const fetchImages = async () => {
    setImagesLoading(true)
    try {
      const res = await fetch("http://localhost:8000/images");
      const data = await res.json();
      setImages(data);
      setImagesLoading(false)
    } catch (err) {
      console.error(err);
      setImagesLoading(false)
    }
  };

  useEffect(() => {
    fetchImages();
  }, []);

  return (
    <Container>
      {loading || imagesLoading? (
        <ImageContainer>
          <Spinner />
        </ImageContainer>
        
      ) : (
      <>
      {modalOpen && (
      <Modal onClick={() => setModalOpen(false)}>
        <CloseButton onClick={() => setModalOpen(false)}>✖</CloseButton>
        {currentIndex > 0 && (
          <NavButton style={{ left: '20px' }} onClick={(e) => {
            e.stopPropagation();
            setCurrentIndex(currentIndex - 1);
          }}>
            ‹
          </NavButton>
        )}
        <ModalImage src={images[currentIndex]} />
        {currentIndex < images.length - 1 && (
          <NavButton style={{ right: '20px' }} onClick={(e) => {
            e.stopPropagation();
            setCurrentIndex(currentIndex + 1);
          }}>
            ›
          </NavButton>
        )}
      </Modal>
    )}


      <ResultsPhotoScroller
            images={images}
            onImageClick={(i) => {
              setCurrentIndex(i);
              setModalOpen(true);
            }}
          />
                <h1>Attendance Register</h1>
      <h3 style={{marginTop: '-40px', fontSize: '30px'}}>{people.length} attendees</h3>
      {people.map((person, index) => (
        <ImageContainer key={index}>
          <PersonDetails>
            <EditableName initialName={"Person " + (index + 1)} />
            
            {person.face ? (
              <FaceImage src={`data:image/jpeg;base64,${person.face}`} alt={`Face`} />
            ) : (
              <FaceImage src={`src/assets/missing_face.JPG`} alt={`Missing Face`} />
            )}
          </PersonDetails>
          <PersonDetails>
            <h2 style={{display: 'flex', alignContent: 'start'}}>Attendee Colour Data</h2>
            <WardrobeDisplay wardrobe={person.description} />
          </PersonDetails>
          <AppearanceScroll>
            <h2 style={{display: 'flex', alignContent: 'start'}}>Attendee Appearances</h2>
            <ScrollItems>
            {person.appearances.map((app, i) => (
              <AppearanceFigure key={i}>
                <ScrollImage 
                  src={`data:image/jpeg;base64,${app.image}`}
                  alt={`Appearance ${app.photo_num}`}
                  height="150"
                />
                <figcaption>Figure {i + 1}/{person.appearances.length}</figcaption>
                <figcaption style={{fontWeight: `bold`}}>(photo {app.photo_num})</figcaption>
              </AppearanceFigure>
            ))}
            </ScrollItems>
          </AppearanceScroll>
        </ImageContainer>
      ))}
      </>)}
    <FetchButton onClick={() => setReady(false)}>Take attendance for another event</FetchButton>
    </Container>
  );
}

export default Results;
