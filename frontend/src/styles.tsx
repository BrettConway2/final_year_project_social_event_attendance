import styled, { keyframes } from "styled-components";

export type Appearance = { photo_num: number; image: string };
export type Person = { description: string; appearances: Appearance[] };


export const Container = styled.div`
  background-color: #e3e3e3;
  color: rgb(16, 42, 68);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  gap: 20px;  
`;

export const FetchButton = styled.button`
  padding: 0.5rem 1rem;
  background-color:rgb(19, 122, 10);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;

  &:hover {
    background-color:rgb(38, 169, 26);
  }
`;


export const FileInputWrapper = styled.div`
  position: relative;
  display: inline-block;
`;

export const HiddenInput = styled.input.attrs({ type: "file" })`
  display: none;
`;

export const StyledLabel = styled.label`
  background-color:rgb(16, 42, 68);
  color: white;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;

  &:hover {
    background-color: #0056b3;
  }
`;


export const spin = keyframes`
  to { transform: rotate(360deg); }
`

export const Spinner = styled.div`
  width: 100px;
  height: 100px;
  border: 9px solid rgba(0,0,0,0.1);
  border-left-color:rgb(30, 66, 79);
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`

export const Overlay = styled.div`
  position: fixed;      /* cover the viewport */
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  display: flex;        /* center children */
  justify-content: center;
  align-items: center;
  z-index: 9999;        /* sit on top of everything else */
`
export const FigImage = styled.img`
  height: 150px;
  flex-shrink: 0;
  border-radius: 8px;
  border: 1px solid #ccc;
`;

export const ImgWrapper = styled.div`
  position: relative;
  display: inline-block;

  &:hover button {
    display: block;
  }
`;

export const DeleteButton = styled.button`
  position: absolute;
  top: 5px;
  right: 5px;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  border: none;
  border-radius: 50%;
  padding: 5px 8px;
  font-size: 14px;
  cursor: pointer;
  display: none; /* Hidden by default */

  &:hover {
    background-color: red;
  }
`;

export const AppearanceScroll = styled.div`
  display: flex;
  overflow-x: auto;
  flex: 1;
  gap: 10px;
  padding-bottom: 10px;
  scrollbar-width: thin;
  scrollbar-color: #888 transparent;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 10px;
  align-items: space-between;
  flex-direction: column;

  &::-webkit-scrollbar {
    height: 8px;
  }
  &::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
  }
`;

export const Modal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 999;
`;

export const ModalImage = styled.img`
  max-width: 90%;
  max-height: 90%;
  border-radius: 10px;
`;

export const NavButton = styled.button`
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: transparent;
  border: none;
  color: white;
  font-size: 2rem;
  cursor: pointer;
  z-index: 1000;
`;

export const CloseButton = styled.button`
  position: absolute;
  top: 20px;
  right: 30px;
  font-size: 2rem;
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  z-index: 1000;
`;




export const ImageContainer = styled.div`
  display: flex;
  flex-direction: row;
  align-items: stretch;
  gap: 10px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  width: 100%;
  height: 100%;
  overflow-x: auto;

`;

export const PersonDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex: 1;
  width: 400px;
  padding: 10px;
  align-items: center;
  border: 1px solid #ccc;
  border-radius: 8px;
`;

export const AppearanceFigure = styled.figure`
  margin: 0;
  text-align: center;
  border-radius: 8px;
`;

export const FaceImage = styled.img`
  width: 200px;
  height: auto;
  border-radius: 8px;
  border: 1px solid #ccc;
  margin-bottom: 10px;
`;

export const ScrollImage = styled(FigImage)`
  flex-shrink: 0;
  min-width: 69px;
`;

export const ScrollItems = styled.div`
  display: flex;
  gap: 10px;
`;

export const Switch = styled.label`
  position: relative;
  display: inline-block;
  width: 50px;
  height: 28px;
`;

export const Input = styled.input`
  opacity: 0;
  width: 0;
  height: 0;
`;

export const Slider = styled.span<{ checked: boolean }>`
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: ${props => (props.checked ? '#4CAF50' : '#ccc')};
  transition: 0.4s;
  border-radius: 34px;

  &::before {
    position: absolute;
    content: "";
    height: 20px; width: 20px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: 0.4s;
    border-radius: 50%;
    transform: ${props => (props.checked ? 'translateX(22px)' : 'translateX(0)')};
  }
`;


export const ScrollContainer = styled.div`
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
  background-color: rgb(16, 42, 68);

  &::-webkit-scrollbar {
    height: 8px;
  }
  &::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
  }
`;

export const Img = styled.img`
  height: 120px;
  flex-shrink: 0;
  border-radius: 5px;
  border: 1px solid white;
`;

export const ImageCard = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  color: white;
  font-size: 14px;
`;

export const SectionHeader = styled.div`
  display: flex;
  font-weight: bold;
  justify-content: flex-start;
  font-size: 30px;
  margin-top: -10px;
  color: white;
`;

