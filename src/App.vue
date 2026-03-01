<script setup lang="ts">
import { ref, onMounted } from 'vue';
import StickerMap from './components/StickerMap.vue'

const squareMap = "/img/map/stickers_square.png";
const rectMap = "/img/map/stickers_rect.png";
const squareCards = "/data/stickers_square.json";
const rectCards = "/data/stickers_rect.json";
const mapImage = ref(squareMap);

// define the types for the rectangles
interface Rectangle {
  id: number;
  name: string;
  bbox: [[number, number], [number, number]];
}



const targets = ref<Rectangle[]>([]);
const useSquare = ref(true);

const loadRectangles = async (jsonPath: string) => {
  try {
    const response = await fetch(jsonPath);
    if (!response.ok) {
      throw new Error(`Failed to load rectangles from ${jsonPath}: ${response.statusText}`);
    }
    const data = await response.json();
    const rectangles: Rectangle[] = data.map((rect: any, index: number) => ({
      id: index,
      name: `/img/card/${rect.name}` || `Rectangle ${index}`,
      bbox: rect.bbox,
    }));
    console.log(`Loaded ${rectangles.length} rectangles from ${jsonPath}`);
    targets.value = rectangles;
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
};

const currentCard = ref<string | null>(null);

const zoomRequested = (index: number) => {
  console.log('Zoom requested for rectangle:', index);
  const target = targets.value[index];
  if (!target || !target.name) {
    console.error('No target found for index:', index);
    return;
  }
  currentCard.value = target.name;
};

const clearZoom = () => {
  console.log('Clearing zoom, hiding card');
  currentCard.value = null;
}

onMounted(() => {
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  useSquare.value = windowWidth <= windowHeight;
  try {
  if (useSquare.value) {
    loadRectangles(squareCards);
    mapImage.value = squareMap;
  } else {
    loadRectangles(rectCards);
    mapImage.value = rectMap;
  }
  } catch (error) {
    console.error('Error loading rectangles:', error);
  }
});

</script>

<template>
  <div class="stickerFrame">
  <StickerMap :mapImage="mapImage" :cardImage="currentCard" :rectangles="targets" 
  @open="zoomRequested" @close="clearZoom" :isSquare="useSquare"
  class="stickerMap"/>
  </div>
</template>

<style scoped>

.stickerFrame {
  position: relative;
  /* width: 100%; */
  /* height: 100%;*/
  height:100%;
  max-height: 80vh;
  margin-left: auto;
  margin-right: auto;
  overflow: hidden;
}

.stickerMap {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  /* height: 100%;*/
  height:auto;
  z-index: 1;
  object-fit: contain;
}

.stickerCard {
  position: absolute;
  top: 0;
  left: 0;
  /*
  transform: translate(-50%, 0);
  */
  height: 100%;
  width: 100%;
  z-index: 10;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}


</style>
