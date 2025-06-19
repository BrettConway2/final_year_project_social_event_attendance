import { ImageCard, Img, ImgWrapper, ScrollContainer, SectionHeader } from "./styles";



export function ResultsPhotoScroller({
  images,
  onImageClick,
}: {
  images: string[];
  onImageClick: (index: number) => void;
}) {
  return (
    <ScrollContainer>
      <div style={{ display: `flex`, flexDirection: `column` }}>
        <SectionHeader>Event Photos</SectionHeader>
        <div style={{ display: `flex`, flexDirection: `row` }}>
          {images.length !== 0 ? (
            images.map((src, i) => (
              <ImgWrapper key={i}>
                <ImageCard>
                  <Img
                    src={src}
                    alt={`Photo ${i + 1}`}
                    onClick={() => onImageClick(i)}
                    style={{ cursor: "pointer" }}
                  />
                  <div style={{fontWeight: `bold`}}>Photo {i + 1}</div>
                </ImageCard>
              </ImgWrapper>
            ))
          ) : (
            <h2 style={{ color: "white" }}>No images uploaded yet</h2>
          )}
        </div>
      </div>
    </ScrollContainer>
  );
}
