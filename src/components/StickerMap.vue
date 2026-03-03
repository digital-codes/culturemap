<template>
    <div class="image-container" ref="container">
        <img :src="mapImage" :class="['main-image', { 'square': isSquare }]" @load="onImageLoad" @click="handleMapClick"
            ref="mapImageRef" />
        <div v-if="cardImage" class="popover" :style="popoverStyle">
            <button class="close-button" @click="closePopover">X</button>
            <img :src="cardImage" class="popover-image" ref="cardImageRef" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';

interface Rectangle {
  bbox: [[number, number], [number, number]];
  // Add other properties if needed
}

const props = defineProps({
    mapImage: {
        type: String,
        required: true,
    },
    cardImage: {
        type: String as () => string | null,
        default: null,
    },
    isSquare: {
        type: Boolean,
        default: false,
    },
    rectangles: {
        type: Array as () => Rectangle[],
        required: true,
    },

});

const emit = defineEmits(["open", "close", "next","prev"]);

const container = ref<HTMLElement | null>(null);
const mapImageRef = ref<HTMLImageElement | null>(null);
const cardImageRef = ref<HTMLImageElement | null>(null);
const imageWidth = ref<number>(0);
const imageHeight = ref<number>(0);

watch (() => props.cardImage, async (newVal) => {
    if (newVal && isMobile()) {
        console.log('Adding touch event listeners for mobile');
        await nextTick(); // Ensure the DOM is updated before adding listeners
        if (!cardImageRef.value) return; // Double-check ref after DOM update
        cardImageRef.value.addEventListener('touchstart', handleTouchStart);
        cardImageRef.value.addEventListener('touchend', handleTouchEnd);
    } else if (cardImageRef.value) {
        console.log('Removing touch event listeners');
        cardImageRef.value.removeEventListener('touchstart', handleTouchStart);
        cardImageRef.value.removeEventListener('touchend', handleTouchEnd);
    }
});

const closePopover = (): void => {
    console.log('Closing popover');
    /*
    if (cardImageRef.value && isMobile()) {
        cardImageRef.value.removeEventListener('touchstart', handleTouchStart);
        cardImageRef.value.removeEventListener('touchend', handleTouchEnd);
    }
    */
    emit('close');
};


const popoverStyle = computed((): { height: string; width: string } => {
    if (!container.value) return { height: '0px', width: '0px' };
    const containerHeight: number = container.value.offsetHeight;
    return {
        height: `${containerHeight}px`,
        width: `${containerHeight}px`,
    };
});

const onImageLoad = () => {
    if (mapImageRef.value) {
        imageWidth.value = mapImageRef.value.naturalWidth;
        imageHeight.value = mapImageRef.value.naturalHeight;
        //imageHeight.value = mapImageRef.value.clientHeight;
        console.log('Image loaded with dimensions:', imageWidth.value, imageHeight.value);
    }
};

let touchStartX = 0;

const isMobile = (): boolean => {
    const mobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    console.log('Is mobile device:', mobile);
    return mobile;
};


const handleTouchStart = (event: TouchEvent) => {
    console.log('Touch start detected');
    if (!event.touches[0]) return;
    touchStartX = event.touches[0].clientX;
    console.log('Touch start X:', touchStartX);
};

const handleTouchEnd = (event: TouchEvent) => {
    console.log('Touch end detected');  
    console.log('Touch event details:', event);
    if (!event.changedTouches[0]) return;
    const touchEndX = event.changedTouches[0].clientX;
    console.log('Touch end X:', touchEndX);
    const diff = touchStartX - touchEndX;
    const threshold = 50;
    console.log('Touch start X:', touchStartX, 'Touch end X:', touchEndX, 'Difference:', diff);

    if (diff > threshold) {
        // Swiped left
        emit('next');
    } else if (diff < -threshold) {
        // Swiped right
        emit('prev');
    }
};

const handleMapClick = (event:any) => {
    if (!mapImageRef.value) return;
    console.log('Map clicked at:', event.clientX, event.clientY);
    const rect = mapImageRef.value.getBoundingClientRect();
    const scaleX = imageWidth.value / rect.width;
    const scaleY = imageHeight.value / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    console.log('Translated click to image coordinates:', x, y);

    let clickedIndex = -1;

    // Iterate from the end to find the last matching rectangle
    for (let i = props.rectangles.length - 1; i >= 0; i--) {
        const rectangle = props.rectangles[i];
        if (!rectangle) continue;
        const [[bboxX0, bboxY0], [bboxX1, bboxY1]] = rectangle.bbox;
        // console.log(`Checking rectangle ${i} with bbox:`, bboxX0, bboxY0, bboxX1, bboxY1);
        if (
            x >= bboxX0 &&
            x <= bboxX1 &&
            y >= bboxY0 &&
            y <= bboxY1
        ) {
            clickedIndex = i;
            break;
        }
    }

    if (clickedIndex !== -1) {
        emit('open', clickedIndex);
    }
};


</script>

<style scoped>
.image-container {
    position: relative;
    display: inline-block;
}

.main-image {
    display: block;
    max-width: 100%;
}

.main-image.square {
    aspect-ratio: 1/1;
    object-fit: cover;
}

.popover {
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    border: 1px solid #ccc;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    z-index: 10;
}

.popover-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.close-button {
    position: absolute;
    top: 5px;
    right: 5px;
    background: white;
    border: solid 2px red;
    font-size: 16px;
    cursor: pointer;
    z-index: 100;
}

@media screen and (max-width: 1400px) {
  .main-image {
    max-height: 50vh;
    margin-left: auto;
    margin-right: auto;
}
}

</style>
