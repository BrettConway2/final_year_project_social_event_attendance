import './App.css'
import { PhotoScroller } from './Photoscroller';
import { CloseButton, Container, FetchButton, FileInputWrapper, HiddenInput, Input, Modal, ModalImage, NavButton, Overlay, Slider, Spinner, StyledLabel, Switch, type Person } from './styles';
import React, { useEffect, useState } from "react";

type UploadImageProps = {
  refetchImages: () => void;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
};


function UploadImage({ refetchImages, setLoading }: UploadImageProps) {
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setLoading(true)
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/upload-image", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    console.log("Uploaded:", data.filename);

    setLoading(false)

    refetchImages();
  };

  return <FileInputWrapper>
  <HiddenInput id="upload" onChange={handleUpload} />
  <StyledLabel htmlFor="upload">Upload Event Photo(s)</StyledLabel>
</FileInputWrapper>
}



type ToggleProps = {
  toggleValue: boolean,
  setToggleValue: React.Dispatch<React.SetStateAction<boolean>>
}


function Toggle({toggleValue, setToggleValue}: ToggleProps) {
  const toggle = () => setToggleValue(prev => !prev);

  return (
    <Switch>
      <Input type="checkbox" checked={toggleValue} onChange={toggle} />
      <Slider checked={toggleValue} />
    </Switch>
  );
}

type UploadProps = {
  setReady: React.Dispatch<React.SetStateAction<boolean>>;
  setPeople: React.Dispatch<React.SetStateAction<Person[]>>;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
};




function Upload({setReady, setPeople, setLoading }: UploadProps) {
  
  const [images, setImages] = useState<string[]>([]);
  const [useFace, setUseFace] = useState<boolean>(true)
  const [useClothes, setUseClothes] = useState<boolean>(true)

  const[imagesLoading, setImagesLoading] = useState<boolean>(true)
  
  const fetchResults = async () => {
    setLoading(true);
    try {
      await fetchImages();
      const res = await fetch(`http://localhost:8000/data?use_face=${useFace}&use_clothes=${useClothes}`);
      const data = await res.json();
      setPeople(data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
      setReady(true);
    }
  };

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

  const handleDelete = async (filename: string) => {
    const confirmed = window.confirm(`Are you sure you want to delete ${filename}?`);
    if (!confirmed) return;

    setLoading(true);
    try {
      await fetch(`http://localhost:8000/delete-image/${filename}`, {
        method: "DELETE",
      });
      fetchImages();
    } catch (error) {
      console.error("Delete failed:", error);
    } finally {
      setLoading(false);
    }
  };


  useEffect(() => {
    fetchImages();
  }, []);

  const [modalOpen, setModalOpen] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  return (
    <>
      Use facial data: <Toggle toggleValue={useFace} setToggleValue={setUseFace}/>
      Use clothing data: <Toggle toggleValue={useClothes} setToggleValue={setUseClothes}/>
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

    <Container>
        {imagesLoading ?
         <Overlay>
            <Spinner />
          </Overlay> 
          : 
          <PhotoScroller
            images={images}
            onDelete={handleDelete}
            onImageClick={(i) => {
              setCurrentIndex(i);
              setModalOpen(true);
            }}
          />}

      <UploadImage refetchImages={fetchImages} setLoading={setLoading}/>
      <FetchButton onClick={fetchResults}>Go!</FetchButton>
    </Container>
    </>
  );
}

export default Upload;
