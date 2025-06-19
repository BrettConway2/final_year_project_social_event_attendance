import styled from "styled-components";
import { DeleteButton, ImgWrapper } from "./styles";

const ScrollContainer = styled.div`
  display: flex;
  overflow-x: auto;
  gap: 10px;
  padding: 10px;
  max-width: 100%;
  box-sizing: border-box;
  scrollbar-width: thin;
  scrollbar-color: #888 transparent;
  border: 2px solid #1c434b;
  border-radius: 5px;
  border-color: #1c434b;
  background-color:rgb(16, 42, 68);

  &::-webkit-scrollbar {
    height: 8px;
  }
  &::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
  }
`;


const Img = styled.img`
  height: 150px;
  flex-shrink: 0; /* Prevent images from shrinking */
  border-radius: 8px;
  border: 1px solid #1c434b;
  border-radius: 5px;
  border-color:rgb(255, 255, 255);
`;



export function PhotoScroller({
    images,
    onDelete,
    onImageClick,
  }: {
    images: string[];
    onDelete: (filename: string) => void;
    onImageClick: (index: number) => void;
  }) {
    return (
      <ScrollContainer>
        {images.length !== 0 ? (
          images.map((src, i) => {
            const filename = src.split('/').pop();
            return (
              <ImgWrapper key={i}>
                <Img
                  src={src}
                  alt={`Photo ${i + 1}`}
                  onClick={() => onImageClick(i)} // 👈
                  style={{ cursor: 'pointer' }}
                />
                <DeleteButton onClick={() => onDelete(filename!)}>✖</DeleteButton>
              </ImgWrapper>
            );
          })
        ) : (
          <h2 style={{ color: 'white' }}>No images uploaded yet</h2>
        )}
      </ScrollContainer>
    );
  }
