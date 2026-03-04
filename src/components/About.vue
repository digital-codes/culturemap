<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
const { t } = useI18n()

import { JsonForms } from '@jsonforms/vue';
import { vanillaRenderers } from '@jsonforms/vue-vanilla';


const renderers = Object.freeze([
    ...vanillaRenderers,
    // here you can add custom renderers
])


const loggedIn = ref(false);
const token = ref("");

const cardImage = ref("/img/card/2_5g_ovl.png");

const data_usr = ref({
    validated: false,
    passkey: "",
    username: ""
})
const schema_usr = {
    "type": "object",
    "properties": {
        "username": {
            "type": "string"
        },
        "passkey": {
            "type": "string"
        }
    }
}

const uischema_usr = {
    "type": "HorizontalLayout",
    "rule": {
        "effect": "DISABLE",
        "condition": {
            "scope": "#/properties/validated",
            "schema": {
                "const": true
            }
        }
    },
    "elements": [
        {
            "type": "Control",
            "label": t('message.username'),
            "scope": "#/properties/username"
        },
        {
            "type": "Control",
            "label": t('message.passkey'),
            "scope": "#/properties/passkey"
        },
    ],
}

// -----------------------------

const data_item = ref({
    name: "",
    url: "",
    location: "",
    geo_lat: -1000,
    geo_lng: -1000,
    img: "",
    description: "",
    tags: []
})

const schema_item = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "url": {
            "type": "string"
        },
        "location": {
            "type": "string"
        },
        "geo_lat": {
            "type": "number"
        },
        "geo_lng": {
            "type": "number"
        },
        "image": {
            "type": "string"
        },
        "description": {
            "type": "string"
        }
    }
}


const uischema_item = {
"type": "HorizontalLayout",
"elements": [
    {
        "type": "Control",
        "label": t('grouptags.name'),
        "scope": "#/properties/name"
    },
    {
        "type": "Control",
        "label": t('grouptags.description'),
        "scope": "#/properties/description"
    },
    {
        "type": "Control",
        "label": t('grouptags.url'),
        "scope": "#/properties/url"
    },
    {
        "type": "Control",
        "label": t('grouptags.location'),
        "scope": "#/properties/location"
    },
    {
        "type": "Control",
        "label": t('message.image'),
        "scope": "#/properties/image"
    },
    ]
}

// -----------------------------

const data_cat = ref({
    cat_music: false,
    cat_painting: false,
    cat_media_art: false,
    cat_sports: false,
    cat_education: false,
    cat_science: false,
    cat_activism: false,
    cat_cooking: false,
    cat_crafts: false,
    cat_dancing: false,
    cat_gaming: false,
    cat_writing: false,
    cat_general_interest: false
})


const schema_cat = {
    "type": "object",
    "properties": {
        "cat_music": {
            "type": "boolean"
        },
        "cat_painting": {
            "type": "boolean"
        },
        "cat_media_art": {
            "type": "boolean"
        },
        "cat_sports": {
            "type": "boolean"
        },
        "cat_education": {
            "type": "boolean"
        },
        "cat_science": {
            "type": "boolean"
        },
        "cat_activism": {
            "type": "boolean"
        },
        "cat_cooking": {
            "type": "boolean"
        },
        "cat_crafts": {
            "type": "boolean"
        },
        "cat_dancing": {
            "type": "boolean"
        },
        "cat_gaming": {
            "type": "boolean"
        },
        "cat_writing": {
            "type": "boolean"
        },
        "cat_general_interest": {
            "type": "boolean"
        }
    }
}


const uischema_cat = {
    "type": "HorizontalLayout",
    "elements": [
        {
            "type": "Control",
            "label": t('grouptags.music'),
            "scope": "#/properties/cat_music"
        },
        {
            "type": "Control",
            "label": t('grouptags.painting'),
            "scope": "#/properties/cat_painting"
        },
        {
            "type": "Control",
            "label": t('grouptags.media_art'),
            "scope": "#/properties/cat_media_art"
        },
        {
            "type": "Control",
            "label": t('grouptags.sports'),
            "scope": "#/properties/cat_sports"
        },
        {
            "type": "Control",
            "label": t('grouptags.education'),
            "scope": "#/properties/cat_education"
        },
        {
            "type": "Control",
            "label": t('grouptags.science'),
            "scope": "#/properties/cat_science"
        },
        {
            "type": "Control",
            "label": t('grouptags.activism'),
            "scope": "#/properties/cat_activism"
        },
        {
            "type": "Control",
            "label": t('grouptags.cooking'),
            "scope": "#/properties/cat_cooking"
        },
        {
            "type": "Control",
            "label": t('grouptags.crafts'),
            "scope": "#/properties/cat_crafts"
        },
        {
            "type": "Control",
            "label": t('grouptags.dancing'),
            "scope": "#/properties/cat_dancing"
        },
        {
            "type": "Control",
            "label": t('grouptags.gaming'),
            "scope": "#/properties/cat_gaming"
        },
        {
            "type": "Control",
            "label": t('grouptags.writing'),
            "scope": "#/properties/cat_writing"
        },
        {
            "type": "Control",
            "label": t('grouptags.general_interest'),
            "scope": "#/properties/cat_general_interest"
        }
    ]
}

// -----------------------------

const onChangeUsr = async (event) => {
    console.log("Change event: ", event);
    data_usr.value = event.data;
    console.log("New data: ", data_usr.value);
    if (!loggedIn.value) {
        console.log("Need loging first, ignoring changes.");
        token.value = "";
        const passkey = data_usr.value.passkey;
        if (passkey === "") {
            return; // ignore
        }
        try {
            const r = await fetch('/php/llamaLogin.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password: passkey, username: "any" })
            })
            if (r.status === 200) {
                const params = await r.json();
                token.value = params.token;
                loggedIn.value = true;
                data_usr.value.validated = true;
                data_usr.value.username = "";
                data_usr.value.passkey = "";
                return;
            }
            console.error('Error checking passkey:', r.statusText);
            alert("An error occurred while checking the passkey. Please try again later.");
            token.value = "";
            loggedIn.value = false;
            data_usr.value.validated = false;
            return;
        }
        catch (error) {
            console.error('Error checking passkey:', error);
            alert("An error occurred while checking the passkey. Please try again later.");
            return;
        }
    }
}

const onChangeItem = async (event) => {
    console.log("Change event: ", event);
    data_item.value = event.data;
    console.log("New data: ", data_item.value);
}
const onChangeCat = async (event) => {
    console.log("Change event: ", event);
    data_cat.value = event.data;
    console.log("New data: ", data_cat.value);
}

const submit = async () => {
    console.log("Submitting data: ", data_item.value, data_cat.value);
    try {
        const r = await fetch('/php/submitGroup.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token.value}`
            },
            body: JSON.stringify({ groupData: data_item.value, categoryData: data_cat.value })
        })
        if (r.status === 200) {
            alert("Group submitted successfully!");
            return;
        }
        console.error('Error submitting group:', r.statusText);
        alert("An error occurred while submitting the group. Please try again later.");
        return;
    }
    catch (error) {
        console.error('Error submitting group:', error);
        alert("An error occurred while submitting the group. Please try again later.");
        return;
    }
}

</script>

<template>
    <div class="about">
        <p>{{ $t('message.about') }}</p>
        <img v-if="loggedIn" :src="cardImage" alt="Group Image" style="max-width: 200px; max-height: 200px; margin-bottom: 1rem;" />
        <button v-if="loggedIn" class="submit" @click="submit">{{ $t('message.submit') }}</button>
        <json-forms :class="loggedIn? 'login_done' : 'login_pending'" :data="data_usr" :schema="schema_usr" :uischema="uischema_usr" :renderers="renderers"
            @change="onChangeUsr" />
        <json-forms v-if="data_usr.validated" class="items" :data="data_item" :schema="schema_item" :uischema="uischema_item" :renderers="renderers" @change="onChangeItem" />
        <json-forms v-if="data_usr.validated" class="categories" :data="data_cat" :schema="schema_cat" :uischema="uischema_cat" :renderers="renderers"
            @change="onChangeCat" />

    </div>
</template>

<style scoped>
.login_pending {
    border-style: solid;
    border-color: red;
    border-width: 1px;
    padding: .5rem;
}
.login_done {
    border-style: solid;
    border-color: green;
    border-width: 1px;
    padding: .5rem;
}

.items {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-top: .5rem;
}

.categories {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: .5rem;
}

</style>

<style>
.horizontal-layout {
    display: flex;
    flex-direction: row;
    gap: 1rem;
}
.horizontal-layout-item {
    flex: 1;
}
</style>
